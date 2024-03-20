
    ######wflw
    printf "********************** NME of dark pose on wflw ****************************\n"
    CUDA_VISIBLE_DEVICES=0,1,2,3 sh tools/dist_test.sh \
        configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/wflw/hrnetv2_w18_wflw_256x256_dark.py \
        exp/wflw-4.pth \
        4 
    printf "********************** end darkpose on wflw ****************************\n"
    printf "\n"
    
    
    ######coco
    printf "********************** NME of dark pose on coco ****************************\n"
    CUDA_VISIBLE_DEVICES=0,1,2,3 sh tools/dist_test.sh \
        configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/coco_wholebody_face/hrnetv2_w18_coco_wholebody_face_256x256_dark.py \
        exp/coco-4.pth \
        4 
    printf "********************** end darkpose on coco ****************************\n"
    printf "\n"
    
    ######300w
    printf "********************** NME of dark pose on 300w ****************************\n"
    CUDA_VISIBLE_DEVICES=0,1,2,3 sh tools/dist_test.sh \
        configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/300w/hrnetv2_w18_300w_256x256_dark.py \
        exp/300W-4.pth \
        4 
    printf "********************** end darkpose on 300w ****************************\n"
    printf "\n"
    
    
    ######aflw
    printf "********************** NME of dark pose on aflw ****************************\n"
    CUDA_VISIBLE_DEVICES=0,1,2,3 sh tools/dist_test.sh \
        configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/aflw/hrnetv2_w18_aflw_256x256_dark.py \
        exp/aflw-4.pth \
        4 
    printf "********************** end darkpose on aflw ****************************\n"
    printf "\n"
    
    
    ######cofw
    printf "********************** NME of dark pose on cofw ****************************\n"
    CUDA_VISIBLE_DEVICES=0,1,2,3 sh tools/dist_test.sh \
        configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/cofw/hrnetv2_w18_cofw_256x256_dark.py \
        exp/cofw-4.pth \
        4 
    printf "********************** end darkpose on cofw ****************************\n"
    printf "\n"