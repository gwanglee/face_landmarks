def train(list2train, max_epoch=10, batch_size=64, num_threads=4, save_path='./train/model.ckpt'):

    num_samples = len(list2train)

    with slim.arg_scope(net.arg_scope()):

        data_ph = tf.placeholder(tf.float32, [None, 32, 32, 1], name='input')
        ans_ph = tf.placeholder(tf.float32, [None, 1])

        data_aug_ph = augment_intensity(data_ph)
        estims = net.infer(data_aug_ph, is_training=True)

        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.001

        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, len(list2train)/batch_size*5, 0.96, staircase=True)
                # drops for every 5 epochs

        loss = tf.losses.mean_squared_error(ans_ph, estims, scope='mse')
        #loss = tf.losses.softmax_cross_entropy(ans_ph, estims)
        #train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss, global_step=global_step)
        train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

        ema = tf.train.ExponentialMovingAverage(decay=0.999)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        sess = tf.Session()
        sess.run(init_op)

        num_iter = 0
        epoch = 0

        #aug_params = {'flip': True, 'pos': 0.05, 'scale': [0.95, 1.1], 'rot':15}
        aug_params = {'flip': True, 'pos': 0.10, 'scale': [0.90, 1.2], 'rot': 30}

        while epoch < max_epoch:

            input_batch, ans_batch, pos = read_data_thread(list2train, num_iter, batch_size=batch_size,
                                                           expand=0.1, aug_params=aug_params, num_threads=num_threads)

            steps, lr, val_loss, ans_pred, _ = sess.run([global_step, learning_rate, loss, estims, train_op], \
                                              feed_dict={data_ph: input_batch, ans_ph: ans_batch})
            num_iter += 1

            steps = num_iter * batch_size
            epoch = int(steps/num_samples)
            print('(pos %4d) Epoch %d, iter %d : loss=%f. lr=%f' % (pos, epoch, num_iter, val_loss, lr))

            if num_iter % 10 == 0:  # validation by drawing
                tiled = make_tile_image(input_batch, ans_batch, ans_pred,
                                batch_size=batch_size, tile_width=int(batch_size/8), tile_height=8)

                cv2.imshow("validation", tiled)
                cv2.waitKey(10)

        # write save code here
        if save_path:
            path_to_save = save_path
            if os.path.exists(path_to_save):
                rmtree(path_to_save)

            os.makedirs(path_to_save)

            tf.train.write_graph(sess.graph.as_graph_def(), os.path.basename(path_to_save), 'model.pbtxt')
            saver = tf.train.Saver()
            save_path = saver.save(sess, path_to_save)
            print('model saved: %s'%save_path)

            image_save_path = path_to_save + ".jpg"
            cv2.imwrite(image_save_path, tiled)
            cv2.imshow("finished", tiled)
            cv2.waitKey(-1)

        sess.close()

if __name__ == '__main__':
    #test_list_shuffle(train_list)
    #test_read_data(train_list)
    import time

    save_name = '%s'%time.strftime('%m%d_%H%M')
    save_dir = './train/' + save_name + '/' + save_name + '.ckpt'
    train(train_list, max_epoch=64, batch_size=64, num_threads=16, save_path=save_dir)