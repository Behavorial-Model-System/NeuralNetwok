import tensorflow as tf
import numpy as np





train_x = [[0.0,0.0], [0.0,1.0], [1.0, 0.0], [1.0, 1.0]]
#train_y = [0.0, 0.0, 0.0, 1.0]
train_y = [[0.0], [0.0], [0.0], [1.0]]


n_inputs = len(train_x[0])
n_outputs = len(train_y[0])
print(n_outputs)

test_x = train_x
test_y = train_y

n_nodes_hl1 = 5

x = tf.placeholder('float',[None, n_inputs])
y = tf.placeholder('float',[None, n_outputs])

# data is input 
def neural_network_model(data):
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([n_inputs, n_nodes_hl1])),
    'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_outputs])),
    'biases': tf.Variable(tf.random_normal([n_outputs]))}

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']),  hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    output = tf.matmul(l1, output_layer['weights']) + output_layer['biases']

    return output

# x is input
def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    # learning rate default is 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 10
    batch_size = 2
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            
            i= 0
            while(i<len(train_x)):
                start = i
                end = i+batch_size

                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i+= batch_size
            print('Epoch', epoch, 'completed out of ', hm_epochs, 'loss:', epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x:test_x, y:test_y}))

train_neural_network(train_x)