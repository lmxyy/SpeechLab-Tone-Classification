#!/usr/bin/env python3
import logging

logger = logging.getLogger('Solver')
stream = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter(fmt='%(name)s:  %(message)s - %(asctime)s',datefmt = '[%d/%b/%Y %H:%M:%S]')
stream.setLevel(logging.DEBUG)
stream.setFormatter(formatter)
logger.addHandler(stream)

class Sover(object):
    def __init__(self, model, data, **kwargs):
        self.model = model
        self.loss_func = nn.CrossEntropyLoss()
        self.X_train = data.get('X_train')
        self.y_train = data.get('y_train')
        self.X_val = data.get('X_val')
        self.y_val = data.get('y_val')

        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)
    
        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)

        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in kwargs.keys())
            logger.error('Unrecognized arguments %s' % extra)
            raise ValueError('Unrecognized arguments %s' % extra)
        if self.update_rule == 'sgd':
            self.optimizer = torch.optim.SGD(
                params = model.params,
                lr = optim_config.get('lr',0.001),
                momentum = optim_config.get('momentum',0),
                weight_decay = optim_config.get('weight_decay',0)
            )
        elif self.update_rule == 'adam':
            self.optimizer = torch.optim.SGD(
                params = model.params,
                lr = optim_config.get('lr',0.001),
                betas = optim_config.get('betas',(0.9,0.999)),
                eps = optim_config.get('eps',1e-8),
                weight_decay = optim_config.get('weight_decay',0)
            )
        else:
            logger.error('Unrecognized update rule %s' % update_rule)
            raise ValueError('Unrecognized update rule %s' % update_rule)
        
        train_dataset = Data.Tensor(data_tensor = X_train,target_tensor = y_train)
        self.loader = Data.Dataloader(
            dataset = train_dataset,
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = 1
        )
        
    def _step(self,X_batch,y_batch):
        out = self.model(X_batch)
        loss = self.loss_func(out,y)
        self.loss_history.append(loss.data.numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def check_accuracy(X,y = None):
        out = self.model(X)
        prediction = torch.max(out, 1)[1]

        pred_y = prediction.data.numpy().squeeze()

        if y == None:
            return pred_y

        acc = np.mean(y.numpy() == pred_y)
        return acc
            
    def train(self):
        num_train = self.X_train.shape[0]
        total_iterations = self.num_epochs*((num_train+self.batch_size-1)//self.batch_size)
        iteration = 0
        for epoch in range(self.num_epochs):
            for step,X_batch,y_batch in enumerate(self.loader):
                self._step(X_batch,y_batch)
                if self.verbose and iteration % self.print_every == 0:
                    logger.info('(Iteration %d / %d) loss: %f' % (iteration + 1, total_iterations, self.loss_history[-1]))
                iteration += 1
            train_acc = self.check_accuracy(self.X_train,self.y_train) # TODO
            self.train_acc_history.append(train_acc)
            if self.X_val != None:
                val_acc = self.check_accuracy(self.X_val,self.y_val)
                self.val_acc_history.append(val_acc)
            if self.verbose:
                if self.X_val != None:
                    logger.info('(Epoch %d / %d) train acc: %f; val_acc: %f' % (epoch, self.num_epochs, train_acc, val_acc))
                else:
                    logger.info('(Epoch %d / %d) train acc: %f' % (epoch, self.num_epochs, train_acc))
                
