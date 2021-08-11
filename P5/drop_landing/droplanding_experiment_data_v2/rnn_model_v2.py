import tensorflow as tf
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--mode', default='train', help='train or test')
parser.add_argument('--num_epochs', default=1)
parser.add_argument('--batch_size', default=50)
parser.add_argument('--learning_rate', default=0.001)
args = parser.parse_args()


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__(window_size,features_num)
        self.layer1=tf.keras.layers.Conv1D(filters=60, kernel_size=5,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[None, window_size,features_num])
        self.layer2=tf.keras.layers.LSTM(60, return_sequences=True)
        self.layer3=tf.keras.layers.LSTM(60, return_sequences=True)
  #tf.keras.layers.Flatten()
        self.layer4=tf.keras.layers.Dense(60, activation="relu")
        self.layer5=tf.keras.layers.Dense(30)
        self.layer6=tf.keras.layers.Dense(6)        
    def call(self,input):
        x=self.layer1(input)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.layer5(x)
        output=self.layer6(x)
        return output

def train():
    model=MyModel(10,36)
    optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    loss_class=tf.keras.losses.Huber()
    checkpoint=tf.train.Checkpoint(myAwesomeModel=model)
    for epoch in range(epochs):
        for (X,Y) in train_set:
            with tf.GradientTape() as tape:
                Y_pred=model(X)
                loss=loss_class(Y,Y_pred)
            grads=tape.gradient(loss,model.variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads,model.variables))
            if epoch%100==0:
                path=checkpoint.save('./save/model.chpt')
                print('model saved')
            
def test():
    model_to_be_restored=MyModel()
    checkpoint=tf.train.Checkpoint(myAwesomeModel=model_to_be_restored)
    checkpoint.restore(tf.train.latest_checkpoint('./save'))
    y_pred=model_to_be_restored.predict(X)
                  


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    if args.mode == 'test':
        test()
