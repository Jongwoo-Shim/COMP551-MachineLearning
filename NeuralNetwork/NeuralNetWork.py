import numpy as np
from skimage.morphology import label
from skimage.measure import regionprops
from skimage.transform import resize

def loadData(shape):
    x = np.loadtxt("train_x.csv", delimiter=",") # load from text 
    y = np.loadtxt("train_y.csv", delimiter=",") 
    x_test = np.loadtxt("test_x.csv", delimiter=",")
    x = x.reshape(-1, 64, 64) # reshape 
    y = y.reshape(-1, 1)
    x = preProcess(x,shape)
    x = x.reshape((50000, shape**2))
    y = y.reshape(50000)
    return x,y, x_test

def preProcess(x,shape): 
    resized = np.zeros(shape=(x.shape[0],shape,shape))
    for n in range(x.shape[0]):
        t = np.array(x[n])
        t[t != 255] = 0
        t[t == 255] = 1
        label_image = label(t)
        num = 0
        max_region = regionprops(label_image)
        for region in regionprops(label_image):
            r0,c0,r1,c1 = region.bbox
            length = max(abs(r0-r1),abs(c0-c1))
            if length > num:
                num = length
                max_region = region
        r0, c0, r1, c1 = max_region['BoundingBox']
        cropped = t[min(r0,r1):max(r0,r1), min(c0,c1):max(c0,c1)]
        resized[n] = resize(cropped, (shape,shape))
        if n%5000 == 0:
            print("On #: ", n)
    return resized

def encode_labels(y, outputs):
    output = np.zeros(shape = (y.shape[0],outputs))
    for i in range(y.shape[0]):
        output[i, int(y[i]) ] = 1.0
    return output

def convertResult(x):
    output = np.zeros(shape = (10))
    t = np.argmax(x)
    output[t] = 1
    return output

def getData(Classes, shape):
    x,y, x_test = loadData(shape)
    y = encode_labels(y,Classes)
    split = int(x.shape[0] * .8)
    x_test = x[split:,:]
    y_test = y[split:,:]
    x_train = x[:split,:]
    y_train = y[:split,:]
    return x_train, y_train, x_test, y_test

def writeErrorCSV(name, error):
    with open(name, 'w') as a:
            for line in error:
                temp = np.array2string(line)
                a.write(temp)
                a.write("\n")

def testResult(x_test, y_test,w1,w2,b1,b2):
    active = relu_activation(np.dot(x_test, w1) + b1)
    scores = np.dot(active, w2) + b2
    probs = softmax(scores) 
    loss = cross_entropy_softmax_loss_array(probs, y_test)
    return loss

def initializeValues(inputs, hidden_nodes, outputs):
    w1 = np.random.normal(0, 1, [inputs, hidden_nodes])
    w2 = np.random.normal(0, 1, [hidden_nodes, outputs]) 
    b1 = np.zeros((1, hidden_nodes))
    b2 = np.zeros((1, outputs))
    return w1, w2, b1, b2             

def relu_activation(data_array):
    return np.maximum(data_array, 0)

def softmax(output_array):
    logits_exp = np.exp(output_array)
    return logits_exp / np.sum(logits_exp, axis = 1, keepdims = True)



def cross_entropy_softmax_loss_array(softmax_probs_array, y_onehot):
    indices = np.argmax(y_onehot, axis = 1).astype(int)
    predicted_probability = softmax_probs_array[np.arange(len(softmax_probs_array)), indices]
    log_preds = np.log(predicted_probability)
    loss = -1.0 * np.sum(log_preds) / len(log_preds)
    return loss

def regularization_L2_softmax_loss(r_lambda, weight1, weight2):
    weight1_loss = 0.5 * r_lambda * np.sum(weight1 * weight1)
    weight2_loss = 0.5 * r_lambda * np.sum(weight2 * weight2)
    return weight1_loss + weight2_loss

def feedForward(w1,w2,b1,b2,x_train):
    input_layer = np.dot(x_train, w1)
    hidden_layer = relu_activation(input_layer + b1)
    output_layer = np.dot(hidden_layer, w2) + b2
    output_probs = softmax(output_layer)
    return input_layer, hidden_layer, output_layer, output_probs

def errorRate(w1, w2, r_lambda,output_probs, y_train, hidden_layer):
    loss = cross_entropy_softmax_loss_array(output_probs, y_train)
    loss += regularization_L2_softmax_loss(r_lambda, w1, w2)
    output_error_signal = (output_probs - y_train) / output_probs.shape[0]
    error_signal_hidden = np.dot(output_error_signal, w2.T) 
    error_signal_hidden[hidden_layer <= 0] = 0
    return loss, output_error_signal, error_signal_hidden

def feedBack(x_train,w1,w2, b1,b2, hidden_layer, output_error_signal, error_signal_hidden, r_lambda, learning_rate):
    gw1 = np.dot(x_train.T, error_signal_hidden)
    gw2 = np.dot(hidden_layer.T, output_error_signal)
    gb1 = np.sum(error_signal_hidden, axis = 0, keepdims = True)
    gb2 = np.sum(output_error_signal, axis = 0, keepdims = True)
    gw1 += r_lambda * w1                      
    gw2 += r_lambda * w2
    w1 -= learning_rate * gw1
    w2 -= learning_rate * gw2
    b1 -= learning_rate * gb1
    b2 -= learning_rate * gb2
    return w1, w2, b1, b2

def main():
    #Constants
    nodes = 5
    learning_rate = .001
    r_lambda = .01
    error = []
    test = []
    epoch = 50000
    #Initialization
    x_train, y_train, x_test, y_test = getData(10, 28)
    w1, w2, b1, b2 = initializeValues(28**2, nodes, 10)
    #BackPropagation
    for step in range(epoch):
        #feedForward
        input_layer, hidden_layer, output_layer, output_probs = feedForward(w1,w2,b1,b2,x_train)
        #Error Calculation        
        loss, output_error_signal, error_signal_hidden = errorRate(w1, w2, r_lambda,output_probs, y_train, hidden_layer)
        #FeedBack
        w1, w2, b1, b2 = feedBack(x_train,w1,w2, b1,b2, hidden_layer, output_error_signal, error_signal_hidden, r_lambda, learning_rate)
        error.append(loss)
        print ('Train Loss {0}: {1}'.format(step, loss))
        test.append(testResult(x_test, y_test,w1,w2,b1,b2))
    writeErrorCSV("Trainloss.csv", error)
    writeErrorCSV("Testloss.csv", test)


#Initialization
x_train, y_train, x_test, y_test = getData(10, 28)
nodes = 5
learning_rates = [1/i for i in range(1,11)]
r_lambda = .01
error = []
test = []
epoch = 10
w1, w2, b1, b2 = initializeValues(28**2, nodes, 10)
#BackPropagation
learning_rate = learning_rates[0]
for step in range(epoch):
    #feedForward
    input_layer, hidden_layer, output_layer, output_probs = feedForward(w1,w2,b1,b2,x_train)
    #Error Calculation        
    loss, output_error_signal, error_signal_hidden = errorRate(w1, w2, r_lambda,output_probs, y_train, hidden_layer)
    #FeedBack
    w1, w2, b1, b2 = feedBack(x_train,w1,w2, b1,b2, hidden_layer, output_error_signal, error_signal_hidden, r_lambda, learning_rate)
    error.append(loss)
    print ('Train Loss {0}: {1}'.format(step, loss))
    testError = testResult(x_test, y_test,w1,w2,b1,b2)
    print(testError)
    test.append(testError)
print(test)
writeErrorCSV("Trainloss.csv", error)
writeErrorCSV("Testloss.csv", test)
main()