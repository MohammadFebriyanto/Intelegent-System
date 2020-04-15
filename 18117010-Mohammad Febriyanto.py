from sklearn import datasets
from scipy import misc
from sklearn.svm import SVC

digits = datasets.load_digits()
features = digits.data 
labels = digits.target

clf = SVC(degree=4).fit(features, labels)

count = 0
right = 0

for i in range(0,10):    
    for k in range(0,31):
        img = misc.imread('./testing/'+str(i)+'/'+str(k)+'.jpg')
        img = misc.imresize(img, (8,8))
        img = img.astype(digits.images.dtype)
        img = misc.bytescale(img, high=16, low=0)
        count += 1

        
        try:
            x_test = []
            for eachRow in img:
                for eachPixel in eachRow:
                    x_test.append(sum(eachPixel)/3.0)
            if clf.predict([x_test])[0] == i:
                right += 1
        except TypeError : #Menghindari datatest yang rusak
            count += -1

print('accuracy_score = %.5f ' %(right*100/count) +'%' )