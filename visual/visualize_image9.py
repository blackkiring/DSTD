from preprocess.preprocess import *
transformer = torch.load('/home/gpu/Experiment/xl/model/image10/model_image9_7_25.pkl')
transformer.cuda()

# 上色
class_count = np.zeros(Categories_Number)
out_clour = np.zeros((label_row, label_column, 3))
gt_clour = np.zeros((label_row, label_column, 3))
def clour_model(transformer, all_data_loader):
    for step, (ms4, pan, gt_xy) in enumerate(all_data_loader):
        ms4 = ms4.cuda()
        pan = pan.cuda()
        with torch.no_grad():
            output,_,_,_,_= transformer(ms4, pan)
        pred_y = torch.max(output, 1)[1].cuda().data.squeeze()
        pred_y_numpy = pred_y.cpu().numpy()
        gt_xy = gt_xy.numpy()
        for k in range(len(gt_xy)):
            if pred_y_numpy[k] == 0:
                class_count[0] = class_count[0] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [255, 255, 0]
                gt_clour[gt_xy[k][0]][gt_xy[k][1]] = [255, 255, 0]
            elif pred_y_numpy[k] == 1:
                class_count[1] = class_count[1] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [255, 0, 0]
                gt_clour[gt_xy[k][0]][gt_xy[k][1]] = [255, 0, 0]
            elif pred_y_numpy[k] == 2:
                class_count[2] = class_count[2] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [33, 145, 237]
                gt_clour[gt_xy[k][0]][gt_xy[k][1]] = [33, 145, 237]
            elif pred_y_numpy[k] == 3:
                class_count[3] = class_count[3] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [201, 252, 189]
                gt_clour[gt_xy[k][0]][gt_xy[k][1]] = [201, 252, 189]
            elif pred_y_numpy[k] == 4:
                class_count[4] = class_count[4] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [0,0,230]
                gt_clour[gt_xy[k][0]][gt_xy[k][1]] = [0,0,230]
            elif pred_y_numpy[k] == 5:
                class_count[5] = class_count[5] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [0,255,0]
                gt_clour[gt_xy[k][0]][gt_xy[k][1]] = [0,255,0]
            elif pred_y_numpy[k] == 6:
                class_count[6] = class_count[6] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [240,32,160]
                gt_clour[gt_xy[k][0]][gt_xy[k][1]] = [240,32,160]
            elif pred_y_numpy[k] == 7:
                class_count[7] = class_count[7] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [221,160,221]
                gt_clour[gt_xy[k][0]][gt_xy[k][1]] = [221,160,221]
            elif pred_y_numpy[k] == 8:
                class_count[8] = class_count[8] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [140, 230, 240]
                gt_clour[gt_xy[k][0]][gt_xy[k][1]] = [140, 230, 240]
            elif pred_y_numpy[k] == 9:
                class_count[9] = class_count[9] + 1
                out_clour[gt_xy[k][0]][gt_xy[k][1]] = [0, 255, 255]
                gt_clour[gt_xy[k][0]][gt_xy[k][1]] = [0, 255, 255]
            if label_np[gt_xy[k][0]][gt_xy[k][1]] == 255:
                gt_clour[gt_xy[k][0]][gt_xy[k][1]]=[0, 0, 0]
    print(class_count)
    cv2.imwrite("/home/gpu/Experiment/xl/clour_images/Beijing3.png", out_clour)
    cv2.imwrite("/home/gpu/Experiment/xl/clour_images/Beijing_gt.png", gt_clour)
clour_model(transformer,  all_data_loader)
