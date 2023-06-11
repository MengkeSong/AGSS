import os

# root = r'C:\Users\Administrator\Desktop\CPD-master\sal_map\first50%/'
root = r'C:\Users\Administrator\Desktop\CPD-master\sal_map\last50%/'
# root = r'C:\Users\Administrator\Desktop\CPD-master\sal_map\test/'
# imgs = os.listdir(os.path.join(root, 'xianzhu'))
imgs = os.listdir(os.path.join(root, 'buxianzhu'))

for img in imgs:
    f = open(r'C:\Users\Administrator\Desktop\CPD-master\sal_map\for_classify\train\train.txt', 'a')

    f.write('rgb/' + img.replace('.png', '.jpg') + ' ' + 'flo/' + img.replace('.png', '.jpg') + ' ' + 'gt/' + img.replace('.jpg', '.png') + ' ' +'0' +  '\n')
    # f.write('test/buxianzhu' + img.replace('.png', '.jpg') + ' ' + '0' + '\n')

# images = []
# depths = []
# gts = []
# with open('D:\\111111111111111\Sook\sook\classification\DUT\\train/train_rgbd.txt', 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         item = line.strip().split(' ')
#         # item[i+1] = item[i]
#         images.append(item[0])
#         depths.append(item[1])
#         gts.append(item[2])
#     long = len(lines)
#     # print(long)#48
#     for i in range(long):#i:0-47
#         s = depths[0]
#         if i+1 < long:
#             depths[i] = depths[i+1]
#             depths[long-1] = s
# # print(len(images))
# # print(images)
# # print(len(depths))
# # print(depths)
# for i in range(len(depths)):
#     f = open('D:\\111111111111111\Sook\sook\classification\DUT\\train/train_rgbd_trans.txt', 'a')
#     f.write(images[i].replace('.png', '.jpg') + ' '  + depths[i].replace('.jpg', '.png') + ' '  + gts[i].replace('.jpg', '.png') + ' ' + '0' + '\n')