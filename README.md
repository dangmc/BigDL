# BigDL
echo "# BigDL" >> README.md
git init
git add README.md
git commit -m "first commit"
git remote add origin https://github.com/dangmc/BigDL.git
git push -u origin master

from bigdl.util.tf_utils import dump_model
import tensorflow as tf

path = "/home/dangmc/Documents/BigDL/model/modelBigDL"
ckpt = "model/cnn.bin"
graph = "model/cnn.bin.meta"

saver = tf.train.import_meta_graph(graph)
with tf.Session() as sess:
    saver.restore(sess, ckpt)
    dump_model(path=path, sess=sess, ckpt_file=ckpt)
