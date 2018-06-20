# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import numpy as np
import tensorflow as tf

import tfutil

#----------------------------------------------------------------------------
# Convenience func that casts all of its arguments to tf.float32.

def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]

#----------------------------------------------------------------------------
# Extract (ws x ws) patches from a tensor

def extract_patches(matrix, ws, h, w):

    val = []
    for yo in range(ws):
        for xo in range(ws):
            MN = matrix[:, yo:yo+h-ws+1, xo:xo+w-ws+1, :]
            val.append(MN)

    win_ids = tf.stack(val, 3)
    
    return win_ids

#----------------------------------------------------------------------------
# Compute structural loss

def structural_loss(fake1, fake2):

    eps = 1e-4
    win_rad = 1

    win_size = (win_rad * 2 + 1) ** 2
    b, d, h, w = fake1.get_shape().as_list()

    c_h, c_w = h - 2 * win_rad, w - 2 * win_rad
    win_diam = win_rad * 2 + 1

    indsM = tf.reshape(tf.range(h * w), [1, h, w, 1])
    win_ids = extract_patches(indsM, win_rad * 2 + 1, h, w)

    A = tf.reshape(win_ids, [-1, 9, 1])
    A = tf.tile(A, [1, 1, 9])

    col = tf.transpose(A, [0, 2, 1])
    col = tf.cast(tf.reshape(col, [-1]), tf.int64)

    row = tf.cast(tf.reshape(A, [-1]), tf.int64)

    def laplacian_matrix(img):
    
        temp = tf.transpose(img, [0, 2, 3, 1])
        
        winI = extract_patches(temp, win_rad * 2 + 1, h, w)
        winI = tf.reshape(winI, [-1, (h-2)*(w-2), 9, 3])
        
        win_mu = tf.reduce_mean(winI, axis=2, keep_dims=True)
        
        term1 = tf.matmul(tf.transpose(winI, [0,1,3,2]), winI)
        term2 = tf.matmul(tf.transpose(win_mu, [0,1,3,2]), win_mu)
        
        win_var = term1 / win_size - term2
        
        inv = tf.linalg.inv(win_var + (eps/win_size) * tf.eye(3))
        
        X = tf.matmul(winI - win_mu, inv)
        
        vals = tf.eye(win_size) - (1.0/win_size)*(1 + tf.matmul(X, tf.transpose(winI - win_mu, [0,1,3,2])))

        vals = tf.layers.flatten(vals)

        SM = tf.SparseTensor(indices=tf.stack([row, col], 1), values=vals[0], dense_shape=[h*w, h*w])
        
        return SM

    def condition(i, loss):
        return tf.less(i, tf.shape(fake1)[0])

    def action(i, loss):
        
        slice1 = fake1[i:i+1]
        slice2 = fake2[i:i+1]
        
        L1 = laplacian_matrix(slice1)
        L2 = laplacian_matrix(slice2)

        size = fake1.get_shape().as_list()[2]
        
        temp = tf.reshape(slice1, [3, -1])
        covariance = tf.matmul(temp, tf.sparse_tensor_dense_matmul(L2, tf.transpose(temp))) / size**2
        str_loss_1 = tf.trace(covariance)

        temp = tf.reshape(slice2, [3, -1])
        covariance = tf.matmul(temp, tf.sparse_tensor_dense_matmul(L1, tf.transpose(temp))) / size**2
        str_loss_2 = tf.trace(covariance)
        
        str_loss_1 = tf.reshape(str_loss_1, [1,1])
        str_loss_2 = tf.reshape(str_loss_2, [1,1])
        
        loss = tf.concat([loss, str_loss_1, str_loss_2], axis=0)
        
        return tf.add(i, 1), loss

    i = tf.constant(0)
    loss = tf.Variable(np.zeros((0,1), dtype=np.float32))

    final_index, loss = tf.while_loop(condition, action, [i, loss], shape_invariants=[i.get_shape(), tf.TensorShape([None,1])])

    return loss

#----------------------------------------------------------------------------
# Repeat tensors

def tf_repeat(tensor, repeats):
    expanded_tensor = tf.expand_dims(tensor, -1)
    multiples = [1] + repeats
    tiled_tensor = tf.tile(expanded_tensor, multiples = multiples)
    repeated_tensor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
    return repeated_tensor

#----------------------------------------------------------------------------
# Generator loss function used in the paper (WGAN + AC-GAN).

def G_wgan_acgan(G, D, opt, training_set, minibatch_size, real_masks,
    cond_weight = 1.0, **kwargs): # Weight of the conditioning term.

    latents = tf.random_normal([2] + G.input_shapes[0][1:])
    labels = tf.random_uniform([2, 3], minval=-1.0, maxval=1.0)

    repeated_latents = tf_repeat(latents, [4, 1])
    repeated_labels = tf.tile(tf_repeat(labels, [2, 1]), [2, 1])
    repeated_masks = tf.tile(real_masks[:2], [4, 1, 1, 1])

    fake_images_out = G.get_output_for(repeated_latents, repeated_labels, repeated_masks, is_training=True)

    scaled_masks = repeated_masks * 0.5 + 0.5
    masked_colors = tf.reduce_sum(fake_images_out * scaled_masks, axis=[2, 3]) / (tf.reduce_sum(scaled_masks, axis=[2, 3]) + 1e-8)

    fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out, is_training=True))
    loss = -fake_scores_out

    WT = kwargs["weights"]

    if D.output_shapes[1][1] > 0:

        with tf.name_scope('LabelPenalty'):
            label_penalty_fakes = tf.reduce_sum(tf.squared_difference(repeated_labels, fake_labels_out), axis=1, keep_dims=True)
            label_penalty_fakes = tfutil.autosummary('Loss/label_penalty_fakes', label_penalty_fakes)
        loss += label_penalty_fakes * cond_weight

    if WT["generator_color_check"] > 0:
        with tf.name_scope('GeneratorColorCheck'):
            generator_color_check = WT["generator_color_check"] * tf.reduce_sum(tf.squared_difference(repeated_labels, masked_colors), axis=1, keep_dims=True)
            generator_color_check = tfutil.autosummary('Loss/generator_color_check', generator_color_check)
        loss += generator_color_check

    if WT["color_consistency"] > 0:
        with tf.name_scope('ColorConsistency'):
            colors_1, colors_2 = tf.dynamic_partition(masked_colors, [0, 0, 0, 0, 1, 1, 1, 1], 2)
            color_consistency = WT["color_consistency"] * tf.reduce_sum(tf.squared_difference(colors_1, colors_2), axis=1, keep_dims=True)
            color_consistency = tf.tile(color_consistency, [2, 1])
            color_consistency = tfutil.autosummary('Loss/color_consistency', color_consistency)
        loss += color_consistency

    if WT["texture_consistency"] > 0:
        with tf.name_scope('TextureConsistency'):
            fake_images_1, fake_images_2 = tf.dynamic_partition(fake_images_out, [0, 0, 1, 1, 0, 0, 1, 1], 2)
            texture_consistency = WT["texture_consistency"] * structural_loss(fake_images_1, fake_images_2)
            texture_consistency = tfutil.autosummary('Loss/texture_consistency', texture_consistency)
        loss += texture_consistency

    if WT["shape_consistency"] > 0:
        with tf.name_scope('ShapeConsistency'):
            mask_complement = 1.0 - scaled_masks
            masked_difference = tf.reduce_sum(tf.abs(fake_images_out - 1.0) * mask_complement, axis=[1, 2, 3]) / (tf.reduce_sum(mask_complement, axis=[1, 2, 3]) + 1e-8)
            shape_consistency = WT["shape_consistency"] * masked_difference
            shape_consistency = tfutil.autosummary('Loss/shape_consistency', shape_consistency)
        loss += shape_consistency

    return loss

#----------------------------------------------------------------------------
# Discriminator loss function used in the paper (WGAN-GP + AC-GAN).

def D_wgangp_acgan(G, D, opt, training_set, minibatch_size, reals, labels, real_masks,
    wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
    wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
    wgan_target     = 1.0,      # Target value for gradient magnitudes.
    cond_weight     = 1.0,      # Weight of the conditioning terms.
    **kwargs):

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])

    fake_images_out = G.get_output_for(latents, labels, real_masks, is_training=True)
    real_scores_out, real_labels_out = fp32(D.get_output_for(reals, is_training=True))
    fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out, is_training=True))
    real_scores_out = tfutil.autosummary('Loss/real_scores', real_scores_out)
    fake_scores_out = tfutil.autosummary('Loss/fake_scores', fake_scores_out)
    loss = fake_scores_out - real_scores_out

    with tf.name_scope('GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
        mixed_images_out = tfutil.lerp(tf.cast(reals, fake_images_out.dtype), fake_images_out, mixing_factors)
        mixed_scores_out, mixed_labels_out = fp32(D.get_output_for(mixed_images_out, is_training=True))
        mixed_scores_out = tfutil.autosummary('Loss/mixed_scores', mixed_scores_out)
        mixed_loss = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out))
        mixed_grads = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss, [mixed_images_out])[0]))
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
        mixed_norms = tfutil.autosummary('Loss/mixed_norms', mixed_norms)
        gradient_penalty = tf.square(mixed_norms - wgan_target)
    loss += gradient_penalty * (wgan_lambda / (wgan_target**2))

    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = tfutil.autosummary('Loss/epsilon_penalty', tf.square(real_scores_out))
    loss += epsilon_penalty * wgan_epsilon

    if D.output_shapes[1][1] > 0:
        with tf.name_scope('LabelPenalty'):
            label_penalty_reals = tf.reduce_sum(tf.squared_difference(labels, real_labels_out), axis=1)
            label_penalty_fakes = tf.reduce_sum(tf.squared_difference(labels, fake_labels_out), axis=1)
            label_penalty_reals = tfutil.autosummary('Loss/label_penalty_reals', label_penalty_reals)
            label_penalty_fakes = tfutil.autosummary('Loss/label_penalty_fakes', label_penalty_fakes)
        loss += (label_penalty_reals + label_penalty_fakes) * cond_weight
    return loss

#----------------------------------------------------------------------------
