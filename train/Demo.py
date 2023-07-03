from preprocess.preprocess import *
model = ViTLite(img_size=Ms4_patch_size,num_heads=2, mlp_ratio=1, embedding_dim=64, positional_embedding='learnable', num_classes=11).cuda()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15], gamma=0.1)
def train_model(model, train_loader, optimizer, epoch,epochs):
    model.train()
    correct = 0.0

    train_bar=tqdm(train_loader)
    for step, (ms, pan, label, _) in enumerate(train_bar):
        ms, pan, label= ms.cuda(), pan.cuda(), label.cuda()
        optimizer.zero_grad()
        output,xo,yo,x2,y2= model(ms,pan)
        pred_train = output.max(1, keepdim=True)[1]
        correct += pred_train.eq(label.view_as(pred_train).long()).sum().item()
        cosinloss = nn.CosineEmbeddingLoss(margin=0.2)
        loss2=cosinloss(xo,yo,torch.ones(len(label)).cuda())+cosinloss(x2,xo,torch.zeros(len(label)).cuda())+cosinloss(y2,yo,torch.zeros(len(label)).cuda())
        loss=F.cross_entropy(output, label.long())+loss2
        loss.backward()
        optimizer.step()
        train_bar.desc=f"train epoch [{epoch}/{epochs}] loss={loss:.3f} loss2={loss2:.3f}"
    print("Train Accuracy: {:.6f}".format(correct * 100.0 / len(train_loader.dataset)))

test_losses=[]
import time
for epoch in range(1, EPOCH+1):
    train_model(model,  train_loader, optimizer, epoch,EPOCH)
    if epoch==EPOCH:
        start = time.time()
        test_loss=test_model(model,  test_loader)
        end = time.time()
        print('time:{0:f}'.format(end-start))
    scheduler.step()
torch.save(model, 'model.pkl')
