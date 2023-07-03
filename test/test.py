from preprocess.preprocess import *
def test_model(model, test_loader):
    model.eval()
    correct = 0.0
    test_bar=tqdm(test_loader)
    test_metric=np.zeros([Categories_Number,Categories_Number])
    with torch.no_grad():
        for data, data1, target, _  in test_bar:
            data, data1, target= data.cuda(), data1.cuda(), target.cuda()
            output= model(data,data1)
            test_loss = F.cross_entropy(output[0], target.long()).item()
            pred = output[0].max(1, keepdim=True)[1]
            for i in range(len(target)):
                test_metric[int(pred[i].item())][int(target[i].item())]+=1
            correct += pred.eq(target.view_as(pred).long()).sum().item()
        print("test Accuracy:{:.3f} \n".format( 100.0 * correct / len(test_loader.dataset)))
    b=np.sum(test_metric,axis=0)
    accuracy=[]
    c=0
    for i in range(0,Categories_Number):
        a=test_metric[i][i]/b[i]
        accuracy.append(a)
        c+=test_metric[i][i]
        print('category {0:d}: {1:f}'.format(i,a))
    average_accuracy = np.mean(accuracy)
    overall_accuracy=c/np.sum(b,axis=0)
    kappa_coefficient = kappa(test_metric, Categories_Number)
    print('AA: {0:f}'.format(average_accuracy))
    print('OA: {0:f}'.format(overall_accuracy))
    print('KAPPA: {0:f}'.format(kappa_coefficient))
    return 100.0 * correct / len(test_loader.dataset)
def kappa(confusion_matrix, k):
    dataMat = np.mat(confusion_matrix)
    P0 = 0.0
    for i in range(k):
        P0 += dataMat[i, i]*1.0
    xsum = np.sum(dataMat, axis=1)
    ysum = np.sum(dataMat, axis=0)
    Pe  = float(ysum*xsum)/np.sum(dataMat)**2
    OA = float(P0/np.sum(dataMat)*1.0)
    cohens_coefficient = float((OA-Pe)/(1-Pe))
    return cohens_coefficient
model=torch.load('model_Hu4.pkl') #load the model have been trained
test_model(model, test_loader)
