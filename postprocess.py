import cv2
import numpy as np
import os


def post_with_dilate(fas_mask,fas_image,fas_result):
    for i in range(len(fas_image)):
        image_dir = os.path.join(data_dir,fas_image[i])
        mask_dir = os.path.join(data_dir,fas_mask[i])
        result_dir = os.path.join(data_dir,fas_result[i])
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                filename = os.path.splitext(file)[0]
                image_path = os.path.join(image_dir,file)
                mask_path = os.path.join(mask_dir,filename+'-mask.jpg')
                mask = cv2.imread(mask_path,0)
                #the size of dilation
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
                #dilation
                mask_dl = cv2.dilate(mask, kernel)
                #resize the images
                image = cv2.imread(image_path)
                image_resize = cv2.resize(image, (256, 256))
                #normalizate the mask
                mask_dl_bi = mask_dl / 255.
                #stack
                mask_dl_bi_3 = np.empty_like(mask_dl_bi)
                mask_dl_bi_3[:,:,0] = mask_dl_bi
                mask_dl_bi_3[:,:,1] = mask_dl_bi
                mask_dl_bi_3[:,:,2] = mask_dl_bi
                #results
                mult = mask_dl_bi_3 * image_resize
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)
                cv2.imwrite(os.path.join(result_dir, filename + '-m.jpg'), mult)

# def multipy(image_dir,r_dir,mul_dir):
#     file = '0a2f832a47cb581d41ce142ace4bb650.jpg'
#     filename = os.path.splitext(file)[0]
#     mask_dl_path = os.path.join(r_dir,filename+'-mask.jpg')
#     image_path = os.path.join(image_dir,file)
#
#     image = cv2.imread(image_path)
#     image_resize = cv2.resize(image,(256,256))
#     mask_dl = cv2.imread(mask_dl_path)
#     mask_dl_bi = mask_dl/255.
#     mult = mask_dl_bi * image_resize
#     cv2.imwrite(os.path.join(mul_dir,filename+'-m.jpg'),mult)

if __name__=="__main__":
    data_dir = ''
    #path to mask
    fas_mask = ["train_mask/b", "train_mask/d", "train_mask/o",
                "train_mask/s", "train_mask/t", "test_mask/b",
                "test_mask/d", "test_mask/o", "test_mask/s",
                "test_mask/t"]
    #path to image
    fas_image = ["train/Images/blouse", "train/Images/dress", "train/Images/outwear",
                 "train/Images/skirt", "train/Images/trousers", "test/Iamges/blouse",
                 "test/Iamges/dress", "test/Iamges/outwear", "test/Iamges/skirt",
                 "test/Iamges/trousers"]
    #path to  result
    fas_result = ["train_result/b", "train_result/d", "train_result/o",
                 "train_result/s", "train_result/t", "test_result/b",
                 "test_result/d", "test_result/o", "test_result/s",
                 "test_result/t"]

    post_with_dilate(fas_mask, fas_image, fas_result)