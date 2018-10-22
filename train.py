import os
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from model_unreal import SourceModel, TargetModel, Discriminator
import util
import argparse

def read_labeled_image_list(suncg_root_folder, mp3d_root_folder):
    """Reads a .txt file containing pathes and labeles
    Args:
       image_list_file: a .txt file with one /path/to/image per line
       label: optionally, if set label will be pasted after each line
    Returns:
       List with all filenames in file image_list_file
    """
    print("Reading suncg image list...")
    suncg_color_images = []
    suncg_depth_images = []
    for folder in os.listdir(suncg_root_folder):
        for i in range(50):
            color_img = os.path.join(suncg_root_folder, folder, "color%d.png" % i)
            depth_img = os.path.join(suncg_root_folder, folder, "depth%d.png" % i)
            assert(os.path.exists(color_img))
            assert(os.path.exists(depth_img))
            suncg_color_images.append(color_img)
            suncg_depth_images.append(depth_img)

    print("Reading mp3d image list...")
    mp3d_color_images = []
    mp3d_depth_images = []
    for sub_folder in os.listdir(mp3d_root_folder):
        _mp3d_color_images = []
        _mp3d_depth_images = []
        # for folder in os.listdir(os.path.join(mp3d_root_folder, sub_folder)):
        for i in range(50):
            color_img = os.path.join(mp3d_root_folder, sub_folder, "color%d.png" % i)
            depth_img = os.path.join(mp3d_root_folder, sub_folder, "depth%d.png" % i)
            if not (os.path.exists(color_img) and os.path.exists(depth_img)):
                os.system("rm -rf {}".format(os.path.join(mp3d_root_folder, sub_folder)))
                break
            
            _mp3d_color_images.append(color_img)
            _mp3d_depth_images.append(depth_img)
            if i == 49:
                mp3d_color_images += _mp3d_color_images
                mp3d_depth_images += _mp3d_depth_images

    return suncg_color_images, suncg_depth_images, mp3d_color_images, mp3d_depth_images

def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    suncg_color = tf.image.decode_png(tf.read_file(input_queue[0]), channels=3)
    suncg_depth = tf.image.decode_png(tf.read_file(input_queue[1]), channels=1)
    mp3d_color = tf.image.decode_png(tf.read_file(input_queue[2]), channels=3)
    mp3d_depth = tf.image.decode_png(tf.read_file(input_queue[3]), channels=1)
    return suncg_color, suncg_depth, mp3d_color, mp3d_depth

#def preprocess_color(color, input_size):
#    color.set_shape([input_size, input_size, 4])
#    color = tf.to_float(color)
#    return color
#
#def preprocess_depth(depth, input_size):
#    depth.set_shape([input_size, input_size, 1])
#    depth = tf.to_float(depth)
#    return depth
def preprocess_color(color, input_size):
    color = tf.expand_dims(color, 0)
    color = tf.image.resize_bilinear(color, [input_size, input_size])
    color = tf.squeeze(color, axis=0)
    return color

def preprocess_depth(depth, input_size):
    depth = tf.expand_dims(depth, 0)
    depth = tf.image.resize_bilinear(depth, [input_size, input_size])
    depth = tf.squeeze(depth, axis=0)
    return depth

def train(args):
    # Reads pfathes of images together with their labels
    with tf.device("/cpu:0"):
        suncg_color_image_list, suncg_depth_image_list, mp3d_color_image_list, mp3d_depth_image_list = read_labeled_image_list("/home/linchao/minos/gym/suncg", "/home/linchao/minos/gym/matter")
        print(len(suncg_color_image_list), len(suncg_depth_image_list), len(mp3d_color_image_list), len(mp3d_depth_image_list))
        min_len = min(len(suncg_color_image_list), len(suncg_depth_image_list), len(mp3d_color_image_list), len(mp3d_depth_image_list))
        suncg_color_image_list, suncg_depth_image_list, mp3d_color_image_list, mp3d_depth_image_list = suncg_color_image_list[:min_len], suncg_depth_image_list[:min_len], mp3d_color_image_list[:min_len], mp3d_depth_image_list[:min_len]
        
        suncg_color_images = ops.convert_to_tensor(suncg_color_image_list, dtype=tf.string)
        suncg_depth_images = ops.convert_to_tensor(suncg_depth_image_list, dtype=tf.string)
        mp3d_color_images = ops.convert_to_tensor(mp3d_color_image_list, dtype=tf.string)
        mp3d_depth_images = ops.convert_to_tensor(mp3d_depth_image_list, dtype=tf.string)
        # Makes an input queue
        input_queue = tf.train.slice_input_producer([suncg_color_images, suncg_depth_images, mp3d_color_images, mp3d_depth_images],
                                                    num_epochs=args.num_epochs,
                                                    shuffle=True)
        
        suncg_color_image, suncg_depth_image, mp3d_color_image, mp3d_depth_image = read_images_from_disk(input_queue)
        
        # Optional Preprocessing or Data Augmentation
        # tf.image implements most of the standard image augmentation
        suncg_color_image = preprocess_color(suncg_color_image, args.input_size)
        mp3d_color_image = preprocess_color(mp3d_color_image, args.input_size)
        suncg_depth_image = preprocess_depth(suncg_depth_image, args.input_size)
        mp3d_depth_image = preprocess_depth(mp3d_depth_image, args.input_size)
        
        # Optional Image and Label Batching
        suncg_color_batch, suncg_depth_batch, mp3d_color_batch, mp3d_depth_batch = tf.train.batch([suncg_color_image, suncg_depth_image, mp3d_color_image, mp3d_depth_image],
                                              batch_size=args.batch_size)
    
    with tf.device("/gpu:0"):
        # suncg_input = tf.concat([suncg_color_batch, suncg_depth_batch], axis=3)
        # mp3d_input = tf.concat([mp3d_color_batch, mp3d_depth_batch], axis=3)
        suncg_input = suncg_color_batch
        mp3d_input = mp3d_color_batch
        source_input = tf.concat([suncg_input, suncg_input], axis=0)
        target_input = tf.concat([mp3d_input, suncg_input], axis=0)
        source_model = SourceModel(args, source_input)
        target_model = TargetModel(args, target_input)
        suncg_output, suncg_output_source = tf.split(source_model.output, num_or_size_splits=2, axis=0)
        mp3d_output, suncg_output_target = tf.split(target_model.output, num_or_size_splits=2, axis=0)
        adversary_ft = tf.concat([suncg_output, mp3d_output], 0)
        discriminator = Discriminator(args, adversary_ft)
        adversary_logits = discriminator.output
        label_ms = tf.fill([args.batch_size, 1], 1.0)
        label_mt = tf.fill([args.batch_size, 1], 0.0)
        adversary_label = tf.concat([label_ms, label_mt], 0)
        mapping_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = adversary_logits, labels = 1 - adversary_label)
        adversary_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = adversary_logits, labels = adversary_label)
        suncg_output_source = tf.Print(suncg_output_source, [suncg_output_source], message="source")
        suncg_output_target = tf.Print(suncg_output_target, [suncg_output_target], message="target")
        identity_loss = tf.nn.l2_loss(suncg_output_source - suncg_output_target) * args.idt_loss
        
        # trainable_variables = tf.trainable_variables() # target_model, discriminator
        source_vars = list(util.collect_vars('source').values())
        target_vars = list(util.collect_vars('target').values())
        disc_vars = list(util.collect_vars('disc').values())
        target_l2_norm = tf.add_n([tf.nn.l2_loss(v) for v in target_vars]) * args.l2_norm
        disc_l2_norm = tf.add_n([tf.nn.l2_loss(v) for v in disc_vars]) * args.l2_norm
        l2_norm = target_l2_norm + disc_l2_norm
        target_grads = tf.gradients(mapping_loss + target_l2_norm + identity_loss, target_vars, name="target_grads")
        disc_grads = tf.gradients(adversary_loss + disc_l2_norm, disc_vars, name="disc_grads")
        lr_var = tf.Variable(args.lr, name='learning_rate', trainable=False)
        optimizer = tf.train.AdamOptimizer(lr_var) # different from adda
        apply_op = optimizer.apply_gradients(zip(target_grads+disc_grads, target_vars+disc_vars), name='apply_op')
        # apply_target_op = optimizer.apply_gradients(zip(target_grads, target_vars), name='target_apply_op')
        # apply_disc_op = optimizer.apply_gradients(zip(disc_grads, disc_vars), name='disc_apply_op')
        _extra_train_ops = []
        train_op = tf.group([apply_op] + _extra_train_ops)
        m_loss = tf.reduce_mean(mapping_loss)
        a_loss = tf.reduce_mean(adversary_loss)
        weight_norm = tf.reduce_mean(target_l2_norm)+tf.reduce_mean(disc_l2_norm)
        tf.summary.scalar('lr', optimizer._lr)
        tf.summary.scalar('mapping loss', m_loss)
        tf.summary.scalar('adversary loss', a_loss)
        tf.summary.scalar('weight norm', weight_norm)
        tf.summary.scalar('l2 loss', identity_loss)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./tensorboard/data")

    sess = util.get_session()
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    util.load_checkpoints("/home/linchao/unreal/suncg_s/checkpoint-13100068", "net_-1", "source", sess=sess)
    # util.load_variables_redir("/home/fengda/baselines/saved_models/exp_017/checkpoint_10000.pt", 'a2c_model/pi', 'a2c_model/pi', sess=sess)
    # util.load_variables_redir("/home/fengda/baselines/saved_models/exp_017/checkpoint_10000.pt", 'a2c_model1/pi', 'a2c_model/pi', sess=sess)
    tf.train.start_queue_runners(sess)

    cnt = 0
    for epoch in range(args.num_epochs):
        for i_batch in range(int(min_len/args.batch_size)):
            _, summary, _m_loss, _a_loss, _idt_loss, _l2_norm = sess.run([train_op, merged, m_loss, a_loss, identity_loss, l2_norm])
            writer.add_summary(summary, cnt)
            print("{}/{} loss: {} {} {} {}".format(epoch, i_batch, _m_loss, _a_loss, _idt_loss, _l2_norm))
            if cnt % args.save_iter == 0:
                # save_file = os.path.join(args.save_path, "checkpoint_{}.pt".format(cnt))
                print("save model iter {}".format(cnt))
                # util.save_variables(save_file, sess=sess)
                util.save_checkpoints(args.save_path, cnt, sess=sess)
            cnt += 1
            
    # print(sess.run(source_model.output))

def main():
    parser = argparse.ArgumentParser(description='Parse args for adda')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--input_size', type=int, default=84)
    parser.add_argument('--input_channel', type=int, default=3)
    parser.add_argument('--load_source', type=str, default=None)
    parser.add_argument('--load_target', type=str, default=None)
    parser.add_argument('--save_path', type=str, default="./saved_models/exp_001")
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--save_iter', type=int, default=100)
    parser.add_argument('--l2_norm', type=float, default=2.5e-5)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--idt_loss', type=float, default=5e-5)

    args = parser.parse_args()
    
    # if args.load_source:
    #     load_model(source_model, args.load_source)
    # if args.load_target:
    #     load_model(target_model, args.load_target)

    train(args)

if __name__ == '__main__':
    main()
