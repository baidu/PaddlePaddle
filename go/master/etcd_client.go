// Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package master

import (
	"context"
	"time"

	"github.com/PaddlePaddle/Paddle/go/utils/networkhelper"
	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/clientv3/concurrency"
	"github.com/nu7hatch/gouuid"
	log "github.com/sirupsen/logrus"
)

const (
	// DefaultLockPath is the default etcd master lock path.
	DefaultLockPath = "/master/lock"
	// DefaultStatePath is the default etcd key for master state.
	DefaultStatePath = "/master/state"
	// DefaultAddrPath is the default etcd key for master address.
	DefaultAddrPath = "/master/addr"
	// DefaultTrainerPath is the default trainer key prefix.
	DefaultTrainerPath = "/trainer"
)

// EtcdClient is the etcd client that the master uses for fault
// tolerance and service registry.
type EtcdClient struct {
	lockPath  string
	statePath string
	client    *clientv3.Client
	lock      *concurrency.Mutex
}

// NewEtcdClient creates a new EtcdClient.
func NewEtcdClient(endpoints []string, addr string, lockPath, addrPath, statePath string, ttlSec int) (*EtcdClient, error) {
	log.Debugf("Connecting to etcd at %v", endpoints)
	// TODO(helin): gracefully shutdown etcd store. Because etcd
	// store holds a etcd lock, even though the lock will expire
	// when the lease timeout, we need to implement graceful
	// shutdown to release the lock.
	cli, err := clientv3.New(clientv3.Config{
		Endpoints:   endpoints,
		DialTimeout: dialTimeout,
	})
	if err != nil {
		return nil, err
	}

	sess, err := concurrency.NewSession(cli, concurrency.WithTTL(ttlSec))
	if err != nil {
		return nil, err
	}

	lock := concurrency.NewMutex(sess, lockPath)
	// It's fine for the lock to get stuck, in this case we have
	// multiple master servers running (only configured to have
	// one master running, but split-brain problem may cause
	// multiple master servers running), and the cluster management
	// software will kill one of them.
	log.Debugf("Trying to acquire lock at %s.", lockPath)
	err = lock.Lock(context.TODO())
	if err != nil {
		return nil, err
	}
	log.Debugf("Successfully acquired lock at %s.", lockPath)

	put := clientv3.OpPut(addrPath, addr)
	resp, err := cli.Txn(context.Background()).If(lock.IsOwner()).Then(put).Commit()
	if err != nil {
		return nil, err
	}

	if !resp.Succeeded {
		log.Fatal("No longer owns the master lock. Exiting.")
	}

	e := &EtcdClient{
		lockPath:  lockPath,
		statePath: statePath,
		client:    cli,
		lock:      lock,
	}

	return e, nil
}

// Save saves the state into the etcd.
func (e *EtcdClient) Save(state []byte) error {
	ctx := context.TODO()
	put := clientv3.OpPut(e.statePath, string(state))
	resp, err := e.client.Txn(ctx).If(e.lock.IsOwner()).Then(put).Commit()
	if err != nil {
		return err
	}

	if !resp.Succeeded {
		log.Errorln("No longer owns the lock, trying to lock again")
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		err := e.lock.Lock(ctx)
		cancel()
		if err != nil {
			// We lost the master lock and can not acquire
			// it back, it means some other master is
			// already started. We don't want cluster
			// management system to kill the master server
			// who is holding the lock and running
			// correctly. So the most feasible solution is
			// to kill current master server. The current
			// state is not saved, but the trainer's RPC
			// call will fail, so the trainer will retry.
			log.Fatalf("Could not acquire the lock at %s: %v. Exiting.", e.lockPath, err)
		}
		log.Infof("Successfully acquired lock at %s.", e.lockPath)
		return e.Save(state)
	}

	return nil
}

// Load loads the state from etcd.
func (e *EtcdClient) Load() ([]byte, error) {
	ctx := context.TODO()
	get := clientv3.OpGet(e.statePath)

	resp, err := e.client.Txn(ctx).If(e.lock.IsOwner()).Then(get).Commit()
	if err != nil {
		return nil, err
	}

	if !resp.Succeeded {
		log.Errorln("No longer owns the lock, trying to lock and load again.")
		err = e.lock.Lock(context.Background())
		if err != nil {
			return nil, err
		}

		return e.Load()
	}

	kvs := resp.Responses[0].GetResponseRange().Kvs
	if len(kvs) == 0 {
		// No state exists
		return nil, nil
	}

	state := kvs[0].Value
	return state, nil
}

// GetKey gets the value by the specify key.
func GetKey(c *clientv3.Client, key string, timeout time.Duration) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	resp, err := c.Get(ctx, key)
	cancel()
	if err != nil {
		return "", err
	}
	kvs := resp.Kvs
	if len(kvs) == 0 {
		return "", nil
	}
	v := kvs[0].Value
	return string(v), nil
}

// RegisterTrainer will generate a uuid and write as key to etcd for each trainer node.
func RegisterTrainer(c *clientv3.Client, timeout time.Duration) error {
	clientUUID, e := uuid.NewV4()
	if e != nil {
		return e
	}
	trainerIP, e := networkhelper.GetExternalIP()
	if e != nil {
		return e
	}
	// put trainer uniq id with etcd lease
	resp, err := c.Grant(context.TODO(), 5)
	if err != nil {
		log.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	_, err = c.Put(ctx, DefaultTrainerPath+"/"+clientUUID.String(), trainerIP, clientv3.WithLease(resp.ID))
	cancel()
	if err != nil {
		log.Error("put trainer key to etcd error:", err)
		return err
	}

	ch, kaerr := c.KeepAlive(context.TODO(), resp.ID)
	if kaerr != nil {
		log.Errorf("keepalive etcd node error: %v", kaerr)
		return kaerr
	}
	// Eat the keep alive message so etcd will not expire the lease.
	go func(ch <-chan *clientv3.LeaseKeepAliveResponse) {
		ka := <-ch
		log.Debugf("keepalive: %d\n", ka.TTL)
	}(ch)

	return nil
}

// watchKey watches the specify key and send to valChan if there is some event.
func watchKey(c *clientv3.Client, key string, valChan chan<- string) {
	rch := c.Watch(context.Background(), key)
	for wresp := range rch {
		for _, ev := range wresp.Events {
			// if received event is DELETE, the value will be an empty string
			log.Infof("received event %s, %q : %q\n", ev.Type, ev.Kv.Key, ev.Kv.Value)
			valChan <- string(ev.Kv.Value)
		}
	}
}
