#!/usr/bin/env python
"""CEVAE model on IHDP
"""



import edward as ed
import tensorflow as tf

from edward.models import Bernoulli, Normal
from progressbar import ETA, Bar, Percentage, ProgressBar
import os
import sys
from datetime import datetime

from datasets import IHDP, IHDP100, IHDP1000, JOBS, TWINS
from evaluation import Evaluator
import numpy as np
import time
from scipy.stats import sem

from utils import fc_net, get_y0_y1
from argparse import ArgumentParser

# Training output collector
training_output = []
epoch_outputs = []
replication_results = []

parser = ArgumentParser()
parser.add_argument('-dataset', choices=['ihdp', 'ihdp100', 'ihdp1000', 'jobs', 'twins'], default='ihdp', help='Dataset to use')
parser.add_argument('-reps', type=int, default=10, help='Number of replications (for IHDP/IHDP100)')
parser.add_argument('-n_reps', type=int, default=None,
                    help='Number of replications to use for IHDP1000/JOBS (default: all)')
parser.add_argument('-earl', type=int, default=10)
parser.add_argument('-lr', type=float, default=0.001)
parser.add_argument('-opt', choices=['adam', 'adamax'], default='adam')
parser.add_argument('-epochs', type=int, default=100)
parser.add_argument('-print_every', type=int, default=10)
args = parser.parse_args()

args.true_post = True

# ============================================================================
# GPU Configuration
# ============================================================================
# Configure GPU settings for optimized training
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # Dynamically allocate GPU memory
config.gpu_options.per_process_gpu_memory_fraction = 0.9  # Use up to 90% of GPU memory
config.allow_soft_placement = True  # Allow operations to be placed on CPU if GPU is unavailable
config.log_device_placement = False  # Set to True for debugging device placement

# Print GPU information
print('=' * 60)
print('GPU Configuration:')
print('- Allow growth: Enabled')
print('- Memory fraction: 90%')
print('- Soft placement: Enabled')
print('=' * 60)
# ============================================================================

# Select dataset
if args.dataset == 'ihdp':
    dataset = IHDP(replications=args.reps)
    dimx = 25
    num_replications = args.reps
elif args.dataset == 'ihdp100':
    n_reps = args.n_reps if args.n_reps else 100
    dataset = IHDP100(n_replications=n_reps)
    dimx = 25
    num_replications = n_reps
    print('Using IHDP100: {} replications (separate mode)'.format(n_reps))
elif args.dataset == 'ihdp1000':
    n_reps = args.n_reps if args.n_reps else 1000
    dataset = IHDP1000(n_replications=n_reps)
    dimx = 25
    num_replications = n_reps
    print('Using IHDP1000: {} replications (separate mode)'.format(n_reps))
elif args.dataset == 'jobs':
    n_reps = args.n_reps if args.n_reps else 10
    dataset = JOBS(n_replications=n_reps)
    dimx = 17  # JOBS has 17 features
    num_replications = n_reps
    print('Using JOBS: {} replications (separate mode)'.format(n_reps))
elif args.dataset == 'twins':
    n_reps = args.n_reps if args.n_reps else 10
    dataset = TWINS(n_replications=n_reps)
    dimx = 47  # TWINS has 47 features (after removing infant_id columns)
    num_replications = n_reps
    print('Using TWINS: {} replications (separate mode)'.format(n_reps))

# Model save path based on dataset
# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')
    print('Created models/ directory for saving trained models')

# Base model path - will be extended with replication index if needed
model_base_path = 'models/cevae_{}'.format(args.dataset)

scores = np.zeros((num_replications, 3))
scores_test = np.zeros((num_replications, 3))

M = None  # batch size during training
d = 20  # latent dimension
lamba = 1e-4  # weight decay
nh, h = 3, 200  # number and size of hidden layers

# Record start time
start_time = datetime.now()
print('Training started at: {}'.format(start_time.strftime('%Y-%m-%d %H:%M:%S')))

def save_results_to_file(dataset_name, num_replications, start_time, end_time,
                          train_scores, test_scores, args, epoch_outputs_list, has_true_ate=False):
    """Save training results to a formatted file following the template"""
    # Ensure record directory exists
    if not os.path.exists('record'):
        os.makedirs('record')

    # Generate filename
    timestamp = end_time.strftime('%Y%m%d_%H%M%S')
    filename = 'record/{}_separate_{}.txt'.format(dataset_name, timestamp)

    # Calculate duration
    duration = end_time - start_time
    hours, remainder = divmod(duration.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)

    # Calculate statistics
    train_mean = np.mean(train_scores, axis=0)
    train_std = sem(train_scores, axis=0) if num_replications > 1 else np.zeros(3)
    test_mean = np.mean(test_scores, axis=0)
    test_std = sem(test_scores, axis=0) if num_replications > 1 else np.zeros(3)

    # Check if dataset has counterfactuals
    has_cf = test_scores[0, 0] < 10  # ITE/ATE/PEHE are usually >10 when they're actually RMSE placeholders

    with open(filename, 'w', encoding='utf-8') as f:
        f.write('=' * 80 + '\n')
        f.write('CEVAE 实验记录\n')
        f.write('=' * 80 + '\n\n')

        f.write('-' * 80 + '\n')
        f.write('【实验配置】\n')
        f.write('-' * 80 + '\n')
        f.write('数据集:           {}\n'.format(dataset_name.upper()))
        f.write('模式:              separate (每个replication独立训练)\n')
        f.write('Replications:      {}\n'.format(num_replications))
        f.write('Epochs:           {}\n'.format(args.epochs))
        f.write('学习率:           {}\n'.format(args.lr))
        f.write('优化器:           {}\n'.format(args.opt))
        f.write('早停检查频率:      {}\n'.format(args.earl))
        f.write('输出频率:          {}\n'.format(args.print_every))
        f.write('\n')
        f.write('开始时间:          {}\n'.format(start_time.strftime('%Y-%m-%d %H:%M:%S')))
        f.write('结束时间:          {}\n'.format(end_time.strftime('%Y-%m-%d %H:%M:%S')))
        f.write('总耗时:            {}小时 {}分钟 {:.0f}秒\n'.format(int(hours), int(minutes), seconds))
        f.write('-' * 80 + '\n\n')

        f.write('-' * 80 + '\n')
        f.write('【最终结果】\n')
        f.write('-' * 80 + '\n\n')

        f.write('CEVAE model total scores on {}\n\n'.format(dataset_name.upper()))

        if has_cf:
            f.write('训练集:\n')
            f.write('- ITE:  {:.3f} ± {:.3f}\n'.format(train_mean[0], train_std[0]))
            f.write('- ATE:  {:.3f} ± {:.3f}\n'.format(train_mean[1], train_std[1]))
            f.write('- PEHE: {:.3f} ± {:.3f}\n\n'.format(train_mean[2], train_std[2]))

            f.write('测试集:\n')
            f.write('- ITE:  {:.3f} ± {:.3f}\n'.format(test_mean[0], test_std[0]))
            f.write('- ATE:  {:.3f} ± {:.3f}\n'.format(test_mean[1], test_std[1]))
            f.write('- PEHE: {:.3f} ± {:.3f}\n\n'.format(test_mean[2], test_std[2]))
        elif has_true_ate:
            f.write('注意: 此数据集有真实ATE值，但没有反事实标签\n')
            f.write('返回指标: [RMSE占位符, ATE误差, RMSE占位符]\n\n')
            f.write('训练集:\n')
            f.write('- ATE误差:  {:.3f} ± {:.3f}\n\n'.format(train_mean[1], train_std[1]))
            f.write('测试集:\n')
            f.write('- ATE误差:  {:.3f} ± {:.3f}\n\n'.format(test_mean[1], test_std[1]))
        else:
            f.write('注意: 此数据集没有反事实标签，以下指标为事实结果RMSE\n\n')
            f.write('训练集:\n')
            f.write('- RMSE: {:.3f} ± {:.3f}\n\n'.format(train_mean[0], train_std[0]))
            f.write('测试集:\n')
            f.write('- RMSE: {:.3f} ± {:.3f}\n\n'.format(test_mean[0], test_std[0]))

        f.write('-' * 80 + '\n')
        f.write('【每个Replication详细结果】\n')
        f.write('-' * 80 + '\n\n')

        for i, (tr_s, te_s) in enumerate(zip(train_scores, test_scores)):
            f.write('Replication {}/{}:\n'.format(i+1, num_replications))
            if has_cf:
                f.write('  Train - ITE: {:.3f}, ATE: {:.3f}, PEHE: {:.3f}\n'.format(tr_s[0], tr_s[1], tr_s[2]))
                f.write('  Test  - ITE: {:.3f}, ATE: {:.3f}, PEHE: {:.3f}\n'.format(te_s[0], te_s[1], te_s[2]))
            elif has_true_ate:
                f.write('  Train - ATE误差: {:.3f}\n'.format(tr_s[1]))
                f.write('  Test  - ATE误差: {:.3f}\n'.format(te_s[1]))
            else:
                f.write('  Train - RMSE: {:.3f}\n'.format(tr_s[0]))
                f.write('  Test  - RMSE: {:.3f}\n'.format(te_s[0]))
            f.write('\n')

        f.write('-' * 80 + '\n')
        f.write('【训练过程摘要】\n')
        f.write('-' * 80 + '\n\n')

        # Sample some epoch outputs
        sample_epochs = min(10, len(epoch_outputs_list))
        step = len(epoch_outputs_list) // sample_epochs if sample_epochs > 0 else 0

        f.write('关键训练节点输出 (共{}个epoch，显示其中{}个):\n\n'.format(len(epoch_outputs_list), sample_epochs))
        for idx in range(0, len(epoch_outputs_list), max(1, len(epoch_outputs_list) // sample_epochs)):
            f.write('[Epoch {}]\n{}\n'.format(idx + 1, epoch_outputs_list[idx]))

        f.write('-' * 80 + '\n')
        f.write('【模型保存位置】\n')
        f.write('-' * 80 + '\n\n')

        if num_replications > 1:
            f.write('models/cevae_{}/\n'.format(dataset_name))
            f.write('├── cevae_{}_rep001/\n'.format(dataset_name))
            f.write('├── cevae_{}_rep002/\n'.format(dataset_name))
            f.write('├── ...\n')
            f.write('└── cevae_{}_rep{:03d}/\n\n'.format(dataset_name, num_replications))
        else:
            f.write('models/cevae_{}/\n\n'.format(dataset_name))

        f.write('=' * 80 + '\n')

    print('\n' + '=' * 60)
    print('实验结果已保存到: {}'.format(filename))
    print('=' * 60)

    return filename


for i, (train, valid, test, contfeats, binfeats) in enumerate(dataset.get_train_valid_test()):
    print('\nReplication {}/{}'.format(i + 1, num_replications))

    # Determine model path for this replication
    # For multi-replication datasets, save each replication separately
    if num_replications > 1:
        model_path = '{}_rep{:03d}'.format(model_base_path, i + 1)
        print('Model will be saved to: {}'.format(model_path))
    else:
        model_path = model_base_path
        print('Model will be saved to: {}'.format(model_path))

    (xtr, ttr, ytr), (y_cftr, mu0tr, mu1tr) = train
    (xva, tva, yva), (y_cfva, mu0va, mu1va) = valid
    (xte, tte, yte), (y_cfte, mu0te, mu1te) = test

    # Check if dataset has counterfactuals
    has_counterfactuals = (y_cfte is not None and mu0te is not None and mu1te is not None)

    # Get true ATE if available (for datasets like JOBS)
    true_ate = getattr(dataset, 'true_ate', None)

    if has_counterfactuals:
        evaluator_test = Evaluator(yte, tte, y_cf=y_cfte, mu0=mu0te, mu1=mu1te)
    else:
        # For datasets without counterfactuals (like JOBS), pass true_ate if available
        if true_ate is not None:
            print('Dataset has true ATE ({:.4f}), computing ATE error'.format(true_ate))
            evaluator_test = Evaluator(yte, tte, true_ate=true_ate)
        else:
            print('Warning: Dataset has no counterfactuals and no true ATE, only computing RMSE')
            evaluator_test = Evaluator(yte, tte)

    # reorder features with binary first and continuous after
    perm = binfeats + contfeats
    xtr, xva, xte = xtr[:, perm], xva[:, perm], xte[:, perm]

    xalltr, talltr, yalltr = np.concatenate([xtr, xva], axis=0), np.concatenate([ttr, tva], axis=0), np.concatenate([ytr, yva], axis=0)

    if has_counterfactuals:
        evaluator_train = Evaluator(yalltr, talltr, y_cf=np.concatenate([y_cftr, y_cfva], axis=0),
                                    mu0=np.concatenate([mu0tr, mu0va], axis=0), mu1=np.concatenate([mu1tr, mu1va], axis=0))
    else:
        # For datasets without counterfactuals (like JOBS), pass true_ate if available
        if true_ate is not None:
            evaluator_train = Evaluator(yalltr, talltr, true_ate=true_ate)
        else:
            evaluator_train = Evaluator(yalltr, talltr)

    # zero mean, unit variance for y during training
    ym, ys = np.mean(ytr), np.std(ytr)
    ytr, yva = (ytr - ym) / ys, (yva - ym) / ys
    best_logpvalid = - np.inf

    with tf.Graph().as_default():
        sess = tf.InteractiveSession(config=config)

        ed.set_seed(1)
        np.random.seed(1)
        tf.set_random_seed(1)

        x_ph_bin = tf.placeholder(tf.float32, [M, len(binfeats)], name='x_bin')  # binary inputs
        x_ph_cont = tf.placeholder(tf.float32, [M, len(contfeats)], name='x_cont')  # continuous inputs
        t_ph = tf.placeholder(tf.float32, [M, 1])
        y_ph = tf.placeholder(tf.float32, [M, 1])

        x_ph = tf.concat([x_ph_bin, x_ph_cont], 1)
        activation = tf.nn.elu

        # CEVAE model (decoder)
        # p(z)
        z = Normal(loc=tf.zeros([tf.shape(x_ph)[0], d]), scale=tf.ones([tf.shape(x_ph)[0], d]))

        # p(x|z)
        hx = fc_net(z, (nh - 1) * [h], [], 'px_z_shared', lamba=lamba, activation=activation)
        logits = fc_net(hx, [h], [[len(binfeats), None]], 'px_z_bin'.format(i + 1), lamba=lamba, activation=activation)
        x1 = Bernoulli(logits=logits, dtype=tf.float32, name='bernoulli_px_z')

        mu, sigma = fc_net(hx, [h], [[len(contfeats), None], [len(contfeats), tf.nn.softplus]], 'px_z_cont', lamba=lamba,
                           activation=activation)
        x2 = Normal(loc=mu, scale=sigma, name='gaussian_px_z')

        # p(t|z)
        logits = fc_net(z, [h], [[1, None]], 'pt_z', lamba=lamba, activation=activation)
        t = Bernoulli(logits=logits, dtype=tf.float32)

        # p(y|t,z)
        mu2_t0 = fc_net(z, nh * [h], [[1, None]], 'py_t0z', lamba=lamba, activation=activation)
        mu2_t1 = fc_net(z, nh * [h], [[1, None]], 'py_t1z', lamba=lamba, activation=activation)
        y = Normal(loc=t * mu2_t1 + (1. - t) * mu2_t0, scale=tf.ones_like(mu2_t0))

        # CEVAE variational approximation (encoder)
        # q(t|x)
        logits_t = fc_net(x_ph, [d], [[1, None]], 'qt', lamba=lamba, activation=activation)
        qt = Bernoulli(logits=logits_t, dtype=tf.float32)
        # q(y|x,t)
        hqy = fc_net(x_ph, (nh - 1) * [h], [], 'qy_xt_shared', lamba=lamba, activation=activation)
        mu_qy_t0 = fc_net(hqy, [h], [[1, None]], 'qy_xt0', lamba=lamba, activation=activation)
        mu_qy_t1 = fc_net(hqy, [h], [[1, None]], 'qy_xt1', lamba=lamba, activation=activation)
        qy = Normal(loc=qt * mu_qy_t1 + (1. - qt) * mu_qy_t0, scale=tf.ones_like(mu_qy_t0))
        # q(z|x,t,y)
        inpt2 = tf.concat([x_ph, qy], 1)
        hqz = fc_net(inpt2, (nh - 1) * [h], [], 'qz_xty_shared', lamba=lamba, activation=activation)
        muq_t0, sigmaq_t0 = fc_net(hqz, [h], [[d, None], [d, tf.nn.softplus]], 'qz_xt0', lamba=lamba,
                                   activation=activation)
        muq_t1, sigmaq_t1 = fc_net(hqz, [h], [[d, None], [d, tf.nn.softplus]], 'qz_xt1', lamba=lamba,
                                   activation=activation)
        qz = Normal(loc=qt * muq_t1 + (1. - qt) * muq_t0, scale=qt * sigmaq_t1 + (1. - qt) * sigmaq_t0)

        # Create data dictionary for edward
        data = {x1: x_ph_bin, x2: x_ph_cont, y: y_ph, qt: t_ph, t: t_ph, qy: y_ph}

        # sample posterior predictive for p(y|z,t)
        y_post = ed.copy(y, {z: qz, t: t_ph}, scope='y_post')
        # crude approximation of the above
        y_post_mean = ed.copy(y, {z: qz.mean(), t: t_ph}, scope='y_post_mean')
        # construct a deterministic version (i.e. use the mean of the approximate posterior) of the lower bound
        # for early stopping according to a validation set
        y_post_eval = ed.copy(y, {z: qz.mean(), qt: t_ph, qy: y_ph, t: t_ph}, scope='y_post_eval')
        x1_post_eval = ed.copy(x1, {z: qz.mean(), qt: t_ph, qy: y_ph}, scope='x1_post_eval')
        x2_post_eval = ed.copy(x2, {z: qz.mean(), qt: t_ph, qy: y_ph}, scope='x2_post_eval')
        t_post_eval = ed.copy(t, {z: qz.mean(), qt: t_ph, qy: y_ph}, scope='t_post_eval')
        logp_valid = tf.reduce_mean(tf.reduce_sum(y_post_eval.log_prob(y_ph) + t_post_eval.log_prob(t_ph), axis=1) +
                                    tf.reduce_sum(x1_post_eval.log_prob(x_ph_bin), axis=1) +
                                    tf.reduce_sum(x2_post_eval.log_prob(x_ph_cont), axis=1) +
                                    tf.reduce_sum(z.log_prob(qz.mean()) - qz.log_prob(qz.mean()), axis=1))

        inference = ed.KLqp({z: qz}, data)
        optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
        inference.initialize(optimizer=optimizer)

        saver = tf.train.Saver(tf.contrib.slim.get_variables())
        tf.global_variables_initializer().run()

        n_epoch, n_iter_per_epoch, idx = args.epochs, 10 * int(xtr.shape[0] / 100), np.arange(xtr.shape[0])

        # dictionaries needed for evaluation
        tr0, tr1 = np.zeros((xalltr.shape[0], 1)), np.ones((xalltr.shape[0], 1))
        tr0t, tr1t = np.zeros((xte.shape[0], 1)), np.ones((xte.shape[0], 1))
        f1 = {x_ph_bin: xalltr[:, 0:len(binfeats)], x_ph_cont: xalltr[:, len(binfeats):], t_ph: tr1}
        f0 = {x_ph_bin: xalltr[:, 0:len(binfeats)], x_ph_cont: xalltr[:, len(binfeats):], t_ph: tr0}
        f1t = {x_ph_bin: xte[:, 0:len(binfeats)], x_ph_cont: xte[:, len(binfeats):], t_ph: tr1t}
        f0t = {x_ph_bin: xte[:, 0:len(binfeats)], x_ph_cont: xte[:, len(binfeats):], t_ph: tr0t}

        for epoch in range(n_epoch):
            avg_loss = 0.0

            t0 = time.time()
            widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
            pbar = ProgressBar(max_value=n_iter_per_epoch, widgets=widgets)
            pbar.start()
            np.random.shuffle(idx)
            for j in range(n_iter_per_epoch):
                # 只在每 50 次迭代或最后一次时更新进度条，减少刷屏
                if (j + 1) % 50 == 0 or (j + 1) == n_iter_per_epoch:
                    pbar.update(j + 1)
                batch = np.random.choice(idx, 100)
                x_train, y_train, t_train = xtr[batch], ytr[batch], ttr[batch]
                info_dict = inference.update(feed_dict={x_ph_bin: x_train[:, 0:len(binfeats)],
                                                        x_ph_cont: x_train[:, len(binfeats):],
                                                        t_ph: t_train, y_ph: y_train})
                avg_loss += info_dict['loss']

            avg_loss = avg_loss / n_iter_per_epoch
            avg_loss = avg_loss / 100

            if epoch % args.earl == 0 or epoch == (n_epoch - 1):
                logpvalid = sess.run(logp_valid, feed_dict={x_ph_bin: xva[:, 0:len(binfeats)], x_ph_cont: xva[:, len(binfeats):],
                                                            t_ph: tva, y_ph: yva})
                if logpvalid >= best_logpvalid:
                    print('Improved validation bound, old: {:0.3f}, new: {:0.3f}'.format(best_logpvalid, logpvalid))
                    best_logpvalid = logpvalid
                    saver.save(sess, model_path)
                # Always save model at the last epoch to ensure we have a checkpoint to restore
                if epoch == (n_epoch - 1):
                    print('Saving final model at epoch {}'.format(epoch + 1))
                    saver.save(sess, model_path)

            if epoch % args.print_every == 0:
                y0, y1 = get_y0_y1(sess, y_post, f0, f1, shape=yalltr.shape, L=1)
                y0, y1 = y0 * ys + ym, y1 * ys + ym
                score_train = evaluator_train.calc_stats(y1, y0)
                rmses_train = evaluator_train.y_errors(y0, y1)

                y0, y1 = get_y0_y1(sess, y_post, f0t, f1t, shape=yte.shape, L=1)
                y0, y1 = y0 * ys + ym, y1 * ys + ym
                score_test = evaluator_test.calc_stats(y1, y0)

                # Handle datasets without counterfactuals
                rmse_f_tr = rmses_train[0] if rmses_train[1] is not None else score_train[0]
                rmse_cf_tr = rmses_train[1] if rmses_train[1] is not None else score_train[0]

                # Format output based on dataset type
                if true_ate is not None:
                    # JOBS dataset: only ATE error is meaningful
                    epoch_output = "Epoch: {}/{}, log p(x) >= {:0.3f}, ate_err_tr: {:0.3f}, ate_err_te: {:0.3f}, dt: {:0.3f}".format(
                        epoch + 1, n_epoch, avg_loss, score_train[1], score_test[1], time.time() - t0)
                else:
                    # Standard or RMSE-only datasets
                    epoch_output = "Epoch: {}/{}, log p(x) >= {:0.3f}, ite_tr: {:0.3f}, ate_tr: {:0.3f}, pehe_tr: {:0.3f}, " \
                                  "rmse_f_tr: {:0.3f}, rmse_cf_tr: {:0.3f}, ite_te: {:0.3f}, ate_te: {:0.3f}, pehe_te: {:0.3f}, " \
                                  "dt: {:0.3f}".format(epoch + 1, n_epoch, avg_loss, score_train[0], score_train[1], score_train[2],
                                               rmse_f_tr, rmse_cf_tr, score_test[0], score_test[1], score_test[2],
                                               time.time() - t0)
                print(epoch_output)
                epoch_outputs.append(epoch_output)

        saver.restore(sess, model_path)
        y0, y1 = get_y0_y1(sess, y_post, f0, f1, shape=yalltr.shape, L=100)
        y0, y1 = y0 * ys + ym, y1 * ys + ym
        score = evaluator_train.calc_stats(y1, y0)
        scores[i, :] = score

        y0t, y1t = get_y0_y1(sess, y_post, f0t, f1t, shape=yte.shape, L=100)
        y0t, y1t = y0t * ys + ym, y1t * ys + ym
        score_test = evaluator_test.calc_stats(y1t, y0t)
        scores_test[i, :] = score_test

        # Format replication output based on dataset type
        if true_ate is not None:
            rep_output = 'Replication: {}/{}, tr_ate_err: {:0.3f}, te_ate_err: {:0.3f}'.format(
                i + 1, num_replications, score[1], score_test[1])
        else:
            rep_output = 'Replication: {}/{}, tr_ite: {:0.3f}, tr_ate: {:0.3f}, tr_pehe: {:0.3f}' \
                         ', te_ite: {:0.3f}, te_ate: {:0.3f}, te_pehe: {:0.3f}'.format(i + 1, num_replications,
                                                                                score[0], score[1], score[2],
                                                                                score_test[0], score_test[1], score_test[2])
        print(rep_output)
        replication_results.append(rep_output)
        sess.close()

print('CEVAE model total scores on {}'.format(args.dataset.upper()))

# Check if dataset has true ATE (like JOBS)
has_true_ate = hasattr(dataset, 'true_ate') and dataset.true_ate is not None

means = np.mean(scores, axis=0)
if has_true_ate:
    # JOBS dataset: only show ATE error
    if num_replications > 1:
        stds = sem(scores, axis=0)
        print('train ATE error: {:.3f}+-{:.3f}'.format(means[1], stds[1]))
    else:
        print('train ATE error: {:.3f}'.format(means[1]))
else:
    # Standard datasets
    if num_replications > 1:
        stds = sem(scores, axis=0)
        print('train ITE: {:.3f}+-{:.3f}, train ATE: {:.3f}+-{:.3f}, train PEHE: {:.3f}+-{:.3f}' \
              ''.format(means[0], stds[0], means[1], stds[1], means[2], stds[2]))
    else:
        print('train ITE: {:.3f}, train ATE: {:.3f}, train PEHE: {:.3f}' \
              ''.format(means[0], means[1], means[2]))

means = np.mean(scores_test, axis=0)
if has_true_ate:
    # JOBS dataset: only show ATE error
    if num_replications > 1:
        stds = sem(scores_test, axis=0)
        print('test ATE error: {:.3f}+-{:.3f}'.format(means[1], stds[1]))
    else:
        print('test ATE error: {:.3f}'.format(means[1]))
else:
    # Standard datasets
    if num_replications > 1:
        stds = sem(scores_test, axis=0)
        print('test ITE: {:.3f}+-{:.3f}, test ATE: {:.3f}+-{:.3f}, test PEHE: {:.3f}+-{:.3f}' \
              ''.format(means[0], stds[0], means[1], stds[1], means[2], stds[2]))
    else:
        print('test ITE: {:.3f}, test ATE: {:.3f}, test PEHE: {:.3f}' \
              ''.format(means[0], means[1], means[2]))

# Record end time and save results
end_time = datetime.now()
print('Training completed at: {}'.format(end_time.strftime('%Y-%m-%d %H:%M:%S')))

# Save results to file following template format
has_true_ate = hasattr(dataset, 'true_ate') and dataset.true_ate is not None
save_results_to_file(args.dataset, num_replications, start_time, end_time,
                      scores, scores_test, args, epoch_outputs, has_true_ate)
