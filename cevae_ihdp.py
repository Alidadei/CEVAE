#!/usr/bin/env python
"""CEVAE model on IHDP
"""



import tensorflow as tf
import tensorflow_probability as tfp
from progressbar import ETA, Bar, Percentage, ProgressBar

from datasets import IHDP
from evaluation import Evaluator
import numpy as np
import time
from scipy.stats import sem

from utils import fc_net, get_y0_y1
from argparse import ArgumentParser

tfd = tfp.distributions

parser = ArgumentParser()
parser.add_argument('-reps', type=int, default=10)
parser.add_argument('-earl', type=int, default=10)
parser.add_argument('-lr', type=float, default=0.001)
parser.add_argument('-opt', choices=['adam', 'adamax'], default='adam')
parser.add_argument('-epochs', type=int, default=100)
parser.add_argument('-print_every', type=int, default=10)
args = parser.parse_args()

args.true_post = True


dataset = IHDP(replications=args.reps)
dimx = 25
scores = np.zeros((args.reps, 3))
scores_test = np.zeros((args.reps, 3))

M = None  # batch size during training
d = 20  # latent dimension
lamba = 1e-4  # weight decay
nh, h = 3, 200  # number and size of hidden layers

for i, (train, valid, test, contfeats, binfeats) in enumerate(dataset.get_train_valid_test()):
    print('\nReplication {}/{}'.format(i + 1, args.reps))
    (xtr, ttr, ytr), (y_cftr, mu0tr, mu1tr) = train
    (xva, tva, yva), (y_cfva, mu0va, mu1va) = valid
    (xte, tte, yte), (y_cfte, mu0te, mu1te) = test
    evaluator_test = Evaluator(yte, tte, y_cf=y_cfte, mu0=mu0te, mu1=mu1te)

    # reorder features with binary first and continuous after
    perm = binfeats + contfeats
    xtr, xva, xte = xtr[:, perm], xva[:, perm], xte[:, perm]

    xalltr, talltr, yalltr = np.concatenate([xtr, xva], axis=0), np.concatenate([ttr, tva], axis=0), np.concatenate([ytr, yva], axis=0)
    evaluator_train = Evaluator(yalltr, talltr, y_cf=np.concatenate([y_cftr, y_cfva], axis=0),
                                mu0=np.concatenate([mu0tr, mu0va], axis=0), mu1=np.concatenate([mu1tr, mu1va], axis=0))

    # zero mean, unit variance for y during training
    ym, ys = np.mean(ytr), np.std(ytr)
    ytr, yva = (ytr - ym) / ys, (yva - ym) / ys
    best_logpvalid = - np.inf

    # Set random seeds
    np.random.seed(1)
    tf.random.set_seed(1)

    # Define the model using TensorFlow 2.x and TensorFlow Probability
    class CEVAE(tf.keras.Model):
        def __init__(self, d, h, nh, lamba, len_binfeats, len_contfeats):
            super(CEVAE, self).__init__()
            self.d = d
            self.h = h
            self.nh = nh
            self.lamba = lamba
            self.len_binfeats = len_binfeats
            self.len_contfeats = len_contfeats
            self.activation = tf.nn.elu
        
        def encode(self, x, t, y):
            """Encoder: q(z|x,t,y)"""
            # q(t|x)
            logits_t = fc_net(x, [self.d], [[1, None]], 'qt', lamba=self.lamba, activation=self.activation)
            qt = tfd.Bernoulli(logits=logits_t, dtype=tf.float32)
            qt_mean = qt.mean()
            
            # q(y|x,t)
            hqy = fc_net(x, (self.nh - 1) * [self.h], [], 'qy_xt_shared', lamba=self.lamba, activation=self.activation)
            mu_qy_t0 = fc_net(hqy, [self.h], [[1, None]], 'qy_xt0', lamba=self.lamba, activation=self.activation)
            mu_qy_t1 = fc_net(hqy, [self.h], [[1, None]], 'qy_xt1', lamba=self.lamba, activation=self.activation)
            qy = tfd.Normal(loc=qt_mean * mu_qy_t1 + (1. - qt_mean) * mu_qy_t0, scale=tf.ones_like(mu_qy_t0))
            
            # q(z|x,t,y)
            inpt2 = tf.concat([x, qy.mean()], 1)
            hqz = fc_net(inpt2, (self.nh - 1) * [self.h], [], 'qz_xty_shared', lamba=self.lamba, activation=self.activation)
            muq_t0, sigmaq_t0 = fc_net(hqz, [self.h], [[self.d, None], [self.d, tf.nn.softplus]], 'qz_xt0', lamba=self.lamba, activation=self.activation)
            muq_t1, sigmaq_t1 = fc_net(hqz, [self.h], [[self.d, None], [self.d, tf.nn.softplus]], 'qz_xt1', lamba=self.lamba, activation=self.activation)
            qz = tfd.Normal(loc=qt_mean * muq_t1 + (1. - qt_mean) * muq_t0, scale=qt_mean * sigmaq_t1 + (1. - qt_mean) * sigmaq_t0)
            
            return qz, qt, qy
        
        def decode(self, z, t):
            """Decoder: p(x,t,y|z)"""
            # p(x|z)
            hx = fc_net(z, (self.nh - 1) * [self.h], [], 'px_z_shared', lamba=self.lamba, activation=self.activation)
            logits = fc_net(hx, [self.h], [[self.len_binfeats, None]], 'px_z_bin', lamba=self.lamba, activation=self.activation)
            x1 = tfd.Bernoulli(logits=logits, dtype=tf.float32, name='bernoulli_px_z')
            
            mu, sigma = fc_net(hx, [self.h], [[self.len_contfeats, None], [self.len_contfeats, tf.nn.softplus]], 'px_z_cont', lamba=self.lamba, activation=self.activation)
            x2 = tfd.Normal(loc=mu, scale=sigma, name='gaussian_px_z')
            
            # p(t|z)
            logits_t = fc_net(z, [self.h], [[1, None]], 'pt_z', lamba=self.lamba, activation=self.activation)
            pt = tfd.Bernoulli(logits=logits_t, dtype=tf.float32)
            
            # p(y|t,z)
            mu2_t0 = fc_net(z, self.nh * [self.h], [[1, None]], 'py_t0z', lamba=self.lamba, activation=self.activation)
            mu2_t1 = fc_net(z, self.nh * [self.h], [[1, None]], 'py_t1z', lamba=self.lamba, activation=self.activation)
            py = tfd.Normal(loc=t * mu2_t1 + (1. - t) * mu2_t0, scale=tf.ones_like(mu2_t0))
            
            return x1, x2, pt, py
        
        def call(self, x_bin, x_cont, t, y):
            x = tf.concat([x_bin, x_cont], 1)
            qz, qt, qy = self.encode(x, t, y)
            x1, x2, pt, py = self.decode(qz.sample(), t)
            return qz, qt, qy, x1, x2, pt, py
        
        def compute_loss(self, x_bin, x_cont, t, y):
            x = tf.concat([x_bin, x_cont], 1)
            x = tf.cast(x, tf.float32)
            t = tf.cast(t, tf.float32)
            y = tf.cast(y, tf.float32)
            qz, qt, qy = self.encode(x, t, y)
            x1, x2, pt, py = self.decode(qz.sample(), t)
            
            # p(z)
            pz = tfd.Normal(loc=tf.zeros([tf.shape(x)[0], self.d], dtype=tf.float32), scale=tf.ones([tf.shape(x)[0], self.d], dtype=tf.float32))
            
            # Calculate ELBO
            log_p_z = tf.reduce_sum(pz.log_prob(qz.mean()), axis=1, keepdims=True)
            log_q_z = tf.reduce_sum(qz.log_prob(qz.mean()), axis=1, keepdims=True)
            log_p_x1 = tf.reduce_sum(x1.log_prob(x_bin), axis=1, keepdims=True)
            log_p_x2 = tf.reduce_sum(x2.log_prob(x_cont), axis=1, keepdims=True)
            log_p_t = pt.log_prob(t)
            log_p_y = py.log_prob(y)
            log_q_t = qt.log_prob(t)
            log_q_y = qy.log_prob(y)
            
            elbo = tf.reduce_mean(log_p_z - log_q_z + log_p_x1 + log_p_x2 + log_p_t + log_p_y - log_q_t - log_q_y)
            return -elbo
        
        def predict_y(self, x_bin, x_cont, t):
            x = tf.concat([x_bin, x_cont], 1)
            x = tf.cast(x, tf.float32)
            t = tf.cast(t, tf.float32)
            # q(t|x)
            logits_t = fc_net(x, [self.d], [[1, None]], 'qt', lamba=self.lamba, activation=self.activation, reuse=True)
            qt = tfd.Bernoulli(logits=logits_t, dtype=tf.float32)
            qt_mean = qt.mean()
            
            # q(y|x,t)
            hqy = fc_net(x, (self.nh - 1) * [self.h], [], 'qy_xt_shared', lamba=self.lamba, activation=self.activation, reuse=True)
            mu_qy_t0 = fc_net(hqy, [self.h], [[1, None]], 'qy_xt0', lamba=self.lamba, activation=self.activation, reuse=True)
            mu_qy_t1 = fc_net(hqy, [self.h], [[1, None]], 'qy_xt1', lamba=self.lamba, activation=self.activation, reuse=True)
            qy = tfd.Normal(loc=qt_mean * mu_qy_t1 + (1. - qt_mean) * mu_qy_t0, scale=tf.ones_like(mu_qy_t0))
            
            # q(z|x,t,y)
            inpt2 = tf.concat([x, qy.mean()], 1)
            hqz = fc_net(inpt2, (self.nh - 1) * [self.h], [], 'qz_xty_shared', lamba=self.lamba, activation=self.activation, reuse=True)
            muq_t0, sigmaq_t0 = fc_net(hqz, [self.h], [[self.d, None], [self.d, tf.nn.softplus]], 'qz_xt0', lamba=self.lamba, activation=self.activation, reuse=True)
            muq_t1, sigmaq_t1 = fc_net(hqz, [self.h], [[self.d, None], [self.d, tf.nn.softplus]], 'qz_xt1', lamba=self.lamba, activation=self.activation, reuse=True)
            qz = tfd.Normal(loc=qt_mean * muq_t1 + (1. - qt_mean) * muq_t0, scale=qt_mean * sigmaq_t1 + (1. - qt_mean) * sigmaq_t0)
            
            # p(y|t,z)
            mu2_t0 = fc_net(qz.mean(), self.nh * [self.h], [[1, None]], 'py_t0z', lamba=self.lamba, activation=self.activation, reuse=True)
            mu2_t1 = fc_net(qz.mean(), self.nh * [self.h], [[1, None]], 'py_t1z', lamba=self.lamba, activation=self.activation, reuse=True)
            py = tfd.Normal(loc=t * mu2_t1 + (1. - t) * mu2_t0, scale=tf.ones_like(mu2_t0))
            
            return py

    # Create model
    model = CEVAE(d, h, nh, lamba, len(binfeats), len(contfeats))
    optimizer = tf.optimizers.Adam(learning_rate=args.lr)

    # Checkpoint manager
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, 'models', max_to_keep=1)

    n_epoch, n_iter_per_epoch, idx = args.epochs, 10 * int(xtr.shape[0] / 100), np.arange(xtr.shape[0])

    # dictionaries needed for evaluation
    tr0, tr1 = np.zeros((xalltr.shape[0], 1)), np.ones((xalltr.shape[0], 1))
    tr0t, tr1t = np.zeros((xte.shape[0], 1)), np.ones((xte.shape[0], 1))

    # Training loop
    for epoch in range(n_epoch):
        avg_loss = 0.0

        t0 = time.time()
        widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
        pbar = ProgressBar(n_iter_per_epoch, widgets=widgets)
        pbar.start()
        np.random.shuffle(idx)
        for j in range(n_iter_per_epoch):
            pbar.update(j)
            batch = np.random.choice(idx, 100)
            x_train, y_train, t_train = xtr[batch], ytr[batch], ttr[batch]
            x_bin = x_train[:, 0:len(binfeats)]
            x_cont = x_train[:, len(binfeats):]
            
            with tf.GradientTape() as tape:
                loss = model.compute_loss(x_bin, x_cont, t_train, y_train)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            avg_loss += loss.numpy()

        avg_loss = avg_loss / n_iter_per_epoch
        avg_loss = avg_loss / 100

        if epoch % args.earl == 0 or epoch == (n_epoch - 1):
            # Calculate validation loss
            x_bin_val = xva[:, 0:len(binfeats)]
            x_cont_val = xva[:, len(binfeats):]
            val_loss = model.compute_loss(x_bin_val, x_cont_val, tva, yva).numpy()
            logpvalid = -val_loss  # ELBO is negative loss
            if logpvalid >= best_logpvalid:
                print('Improved validation bound, old: {:0.3f}, new: {:0.3f}'.format(best_logpvalid, logpvalid))
                best_logpvalid = logpvalid
                checkpoint_manager.save()

        if epoch % args.print_every == 0:
            # Get predictions for evaluation
            def get_y0_y1(model, x, t0, t1, len_binfeats, shape, L=1):
                y0 = np.zeros(shape, dtype=np.float32)
                y1 = np.zeros(shape, dtype=np.float32)
                x_bin = x[:, 0:len_binfeats]
                x_cont = x[:, len_binfeats:]
                for l in range(L):
                    py0 = model.predict_y(x_bin, x_cont, t0)
                    py1 = model.predict_y(x_bin, x_cont, t1)
                    y0 += py0.mean().numpy() / L
                    y1 += py1.mean().numpy() / L
                return y0, y1

            y0, y1 = get_y0_y1(model, xalltr, tr0, tr1, len(binfeats), yalltr.shape, L=1)
            y0, y1 = y0 * ys + ym, y1 * ys + ym
            score_train = evaluator_train.calc_stats(y1, y0)
            rmses_train = evaluator_train.y_errors(y0, y1)

            y0, y1 = get_y0_y1(model, xte, tr0t, tr1t, len(binfeats), yte.shape, L=1)
            y0, y1 = y0 * ys + ym, y1 * ys + ym
            score_test = evaluator_test.calc_stats(y1, y0)

            print("Epoch: {}/{}, log p(x) >= {:0.3f}, ite_tr: {:0.3f}, ate_tr: {:0.3f}, pehe_tr: {:0.3f}, " \
                  "rmse_f_tr: {:0.3f}, rmse_cf_tr: {:0.3f}, ite_te: {:0.3f}, ate_te: {:0.3f}, pehe_te: {:0.3f}, " \
                  "dt: {:0.3f}".format(epoch + 1, n_epoch, avg_loss, score_train[0], score_train[1], score_train[2],
                                       rmses_train[0], rmses_train[1], score_test[0], score_test[1], score_test[2],
                                       time.time() - t0))

    # Restore best model
    checkpoint.restore(checkpoint_manager.latest_checkpoint)

    # Get final predictions
    def get_y0_y1(model, x, t0, t1, len_binfeats, shape, L=1):
        y0 = np.zeros(shape, dtype=np.float32)
        y1 = np.zeros(shape, dtype=np.float32)
        x_bin = x[:, 0:len_binfeats]
        x_cont = x[:, len_binfeats:]
        for l in range(L):
            py0 = model.predict_y(x_bin, x_cont, t0)
            py1 = model.predict_y(x_bin, x_cont, t1)
            y0 += py0.mean().numpy() / L
            y1 += py1.mean().numpy() / L
        return y0, y1

    y0, y1 = get_y0_y1(model, xalltr, tr0, tr1, len(binfeats), yalltr.shape, L=100)
    y0, y1 = y0 * ys + ym, y1 * ys + ym
    score = evaluator_train.calc_stats(y1, y0)
    scores[i, :] = score

    y0t, y1t = get_y0_y1(model, xte, tr0t, tr1t, len(binfeats), yte.shape, L=100)
    y0t, y1t = y0t * ys + ym, y1t * ys + ym
    score_test = evaluator_test.calc_stats(y1t, y0t)
    scores_test[i, :] = score_test

    print('Replication: {}/{}, tr_ite: {:0.3f}, tr_ate: {:0.3f}, tr_pehe: {:0.3f}' \
          ', te_ite: {:0.3f}, te_ate: {:0.3f}, te_pehe: {:0.3f}'.format(i + 1, args.reps,
                                                                      score[0], score[1], score[2],
                                                                      score_test[0], score_test[1], score_test[2]))

print('CEVAE model total scores')
means, stds = np.mean(scores, axis=0), sem(scores, axis=0)
print('train ITE: {:.3f}+-{:.3f}, train ATE: {:.3f}+-{:.3f}, train PEHE: {:.3f}+-{:.3f}' \
      ''.format(means[0], stds[0], means[1], stds[1], means[2], stds[2]))

means, stds = np.mean(scores_test, axis=0), sem(scores_test, axis=0)
print('test ITE: {:.3f}+-{:.3f}, test ATE: {:.3f}+-{:.3f}, test PEHE: {:.3f}+-{:.3f}' \
      ''.format(means[0], stds[0], means[1], stds[1], means[2], stds[2]))