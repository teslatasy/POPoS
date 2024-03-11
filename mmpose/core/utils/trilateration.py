import cv2
import numpy as np
import math
from mmpose.core.post_processing import transform_preds

def topK_sort(dis, anchor, topK=4):
    # import pdb
    # pdb.set_trace()
    
    dis_new=[]
    anchor_new=[]
    length=len(dis)-1

    sorted_nums = sorted(enumerate(dis), key=lambda x: x[1])
    for i in range(topK):
        dis_new.append( sorted_nums[length-i][1] )
        anchor_new.append( anchor[sorted_nums[length-i][0] ] )
    
    return dis_new, anchor_new

def topK(heatmaps, anchor, topK=[0,5] ):
    N, K, H, W = heatmaps.shape

    anchor_expand = np.zeros( (N, K, H*W, 2) )
    anchor_expand[:,:,] = anchor

    heatmaps_trans = np.reshape(heatmaps, (N,K,-1))
    # heatmaps_trans_map = np.argsort(heatmaps_trans) # zhengxu
    heatmaps_trans_map = np.argsort(-heatmaps_trans) # fanxu
    
    indices = np.indices((N,K,H*W))
    indices[2,:,:,:] = heatmaps_trans_map
    heatmaps_trans_sort = heatmaps_trans[indices[0], indices[1], indices[2]]
    heatmaps_trans_sort=heatmaps_trans_sort[:,:,topK[0]:topK[1]]

    anchor_sort = anchor_expand[indices[0], indices[1], indices[2]]
    anchor_sort = anchor_sort[:,:,topK[0]:topK[1]]

    return heatmaps_trans_sort, anchor_sort

##------------------------------------------
def block_search(hm, center, H, W, step_h=2, step_w=3, gl=False):
    # import pdb
    # pdb.set_trace()
    
    centerX=int(center[0]+0.5)  #这里最大值加了0.5
    centerY=int(center[1]+0.5)
    Ymin=max(centerY-step_h, 0)
    Ymax=min(centerY+step_h+1, H)
    Xmin=max(centerX-step_w, 0)
    Xmax=min(centerX+step_w+1, W)
    
    if gl:
        Ymin=0
        Ymax=H
        Xmin=0
        Xmax=W

    max_val=-99999999.0
    bbox=[Ymin, Ymin+step_h, Xmin, Xmin+step_w]
    sub_prob = hm[Ymin:Ymin+step_h, Xmin:Xmin+step_w]
    sub_anchor=[]

    iter_h = Ymax-Ymin-step_h
    iter_w = Xmax-Xmin-step_w
    for i in range(iter_h):
        ymin= i+Ymin
        ymax= ymin+step_h
        for j in range(iter_w):
            xmin = j+Xmin
            xmax = xmin+step_w
            
            sub_hm_tmp = hm[ymin:ymax, xmin:xmax]
            val = np.sum(sub_hm_tmp)
            # print(f'(i, j)=({i},{j}), [{ymin},{ymax},{xmin},{xmax}], max_val={max_val}')
            if val>=max_val:
                max_val = val
                bbox[0]= ymin
                bbox[1]= ymax
                bbox[2]= xmin
                bbox[3]= xmax

    sub_prob=hm[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    for i in range(bbox[0], bbox[1]):
        for j in range(bbox[2], bbox[3]):
            sub_anchor.append([j, i])
           
            # sub_prob.append(hm[j, i])

    # pdb.set_trace()

    return sub_prob, sub_anchor

def anchor_map(anchor, X, Y):
    # import pdb 
    # pdb.set_trace()
    
    result=[ [ X[pt[0]], Y[pt[1]] ] for pt in anchor ]
    return result

def decode_dis(prob, sigma):
    dis=[]
    for p0 in prob:
        for p in p0:
            p = np.clip(p, 0.000000001, 1.0)
            dis_decode = np.sqrt( -2. * sigma * sigma * np.log(p) )
            # dis_decode =  -2. * sigma * sigma * np.log(p)
            dis.append(dis_decode)

    return dis

# linear least squares
def LLS_solve(anchor, dis):
    # import pdb
    # pdb.set_trace()

    length = len(anchor)-1
    X = np.zeros((length, 2), dtype = np.float32)
    Y = np.zeros((length, 1), dtype = np.float32)

    for idx in range(length):
        X[idx][0] = anchor[length][0] - anchor[idx][0]
        X[idx][1] = anchor[length][1] - anchor[idx][1]
        Y[idx][0] = (dis[idx]**2 + anchor[length][0]**2 + anchor[length][1]**2 - dis[length]**2 - anchor[idx][0]**2 - anchor[idx][1]**2) * 0.5
    
    try:
        ret = np.linalg.inv( np.dot(X.transpose(), X) )
    except:
        tmp=np.dot(X.transpose(), X)  
        print(f'anchor={anchor}')
        print(f'X={X}')
        print(f'Y={Y}')
        print(f'tmp={tmp}')
        print(f'tmp矩阵不存在逆矩阵')

    ret = np.dot(ret, X.transpose() )
    ret = np.dot(ret, Y)

    return [ret[0][0], ret[1][0]]
##------------------------------------------

def get_max4_preds(heatmaps):
    """Get keypoint predictions from score maps.

    Note:
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.

    Returns:
        tuple: A tuple containing aggregated results.

        - preds (np.ndarray[N, K, 4, 2]): Predicted keypoint locations.
        - maxvals (np.ndarray[N, K, 4, 1]): Scores (confidence) of the keypoints.
    """
    assert isinstance(heatmaps,
                      np.ndarray), ('heatmaps should be numpy.ndarray')
    assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    N, K, H, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    indices = np.argsort(heatmaps_reshaped, axis=-1)[:, :, -4:]  # Select top 4 indices # Select top 4 indices
    maxvals = np.take_along_axis(heatmaps_reshaped, indices, axis=-1)
    indices=indices[:,:,::-1]

    preds = np.zeros((N, K, 4, 2), dtype=np.float32)
    preds[:, :, :, 0] = np.tile(np.arange(N * K).reshape(N, K, 1), (1, 1, 4))
    preds[:, :, :, 1] = indices // W
    preds[:, :, :, 0] = indices % W

    return preds,maxvals

def get_max_preds(heatmaps):
    """Get keypoint predictions from score maps. 

    Note:
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.

    Returns:
        tuple: A tuple containing aggregated results.

        - preds (np.ndarray[N, K, 2]): Predicted keypoint location.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    assert isinstance(heatmaps,
                      np.ndarray), ('heatmaps should be numpy.ndarray')
    assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    N, K, _, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W

    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    return preds, maxvals

def fit_gauss_newton(station, anchor, dis):
    iter_max=20
    cost=0
    last_cost=0
    
    x0=station[0]
    y0=station[1]
    
    import pdb
    # pdb.set_trace()

    std = np.var(dis)
    sigma = np.sqrt(std)
    inv_sigma = 1.0/(sigma+0.000000001)
    inv_sigma=1

    for idx in range(iter_max):
        H = np.zeros( (2,2), dtype = np.float32)
        B = np.zeros( (2,1), dtype = np.float32)
        # B = np.zeros( (1,2), dtype = np.float32)
        cost=0
        for idx2 in range(len(anchor)):
            error = (anchor[idx2][0]-x0)**2 + (anchor[idx2][1]-y0)**2 - dis[idx2]**2
            J = np.zeros( (2,1), dtype = np.float32)
            J[0] = -2*(anchor[idx2][0]-x0)
            J[1] = -2*(anchor[idx2][1]-y0)
            H += inv_sigma * inv_sigma * J * J.transpose()
            B += -inv_sigma * inv_sigma * error * J  
            # H += inv_sigma * inv_sigma * J.transpose()*J 
            # B += -inv_sigma * inv_sigma  * J.transpose() * error# 
            # (2,1) doesn't match the broadcast shape (2,2)
            cost += error*error
            # print(f'------idx={idx}, idx2={idx2}, cost={cost}, error={error}')
        
        # pdb.set_trace()
        
        delta = np.linalg.solve(H, B)
        if delta[0] is np.nan:
            print(f'*********error in fit gauss newton*********\n')
            break

        if idx > 0 and cost >= last_cost: 
            
            break
        
        x0 += delta[0]
        y0 += delta[1]
        last_cost = cost
    return [x0, y0]


# # #linear least squares
def keypoints_from_heatmaps_tril(heatmaps, preds, center, scale, sigma=2, phase=False):
    if phase:
        preds, maxvals = get_max_preds(heatmaps)

    N, K, H, W = heatmaps.shape
    x = np.arange(0, W, 1, np.float32)
    y = np.arange(0, H, 1, np.float32)
    for n in range(N):
        for k in range(K):
            heatmap = heatmaps[n][k]
            sub_prob, sub_anchor = block_search(heatmap, preds[n][k], H, W, step_h=2, step_w=2, gl=False )
            sub_dis = decode_dis(sub_prob, sigma=sigma)
            sub_anchor = anchor_map(sub_anchor, x, y)
            station = LLS_solve(sub_anchor, sub_dis)

            preds[n][k][0]=station[0]
            preds[n][k][1]=station[1]
    
    return preds


# # # for test
def keypoints_from_heatmaps_tril_img(img, heatmaps, preds, center, scale, sigma=2, phase=False):
    import pdb
    # pdb.set_trace()

    if phase:
        preds, maxvals = get_max_preds(heatmaps)
    
    N, K, H, W = heatmaps.shape
    x = np.arange(0, W, 1, np.float32)
    y = np.arange(0, H, 1, np.float32)

    anchor_16=[[0,0],[0,0],[0,0],[0,0]]
    dis_16=[0,0,0,0]
    pt_16=[0,0]
    scale=4
    # scale=32 s
    for n in range(N):
        for k in range(K):
            heatmap = heatmaps[n][k]
            sub_prob, sub_anchor = block_search(heatmap, preds[n][k], H, W, step_h=2, step_w=2, gl=False )
            sub_dis = decode_dis(sub_prob, sigma=sigma)
            sub_anchor = anchor_map(sub_anchor, x, y)
            station = LLS_solve(sub_anchor, sub_dis)

            # preds[n][k][0]=station[0]
            # preds[n][k][1]=station[1]
            preds[n][k][0]=station[0]*scale
            preds[n][k][1]=station[1]*scale

            if k==16:
                anchor_16=sub_anchor
                dis_16=sub_dis
                pt_16 = preds[n][k]

    # for i in range(N):
    #     preds[i] = transform_preds(
    #         preds[i], center[i], scale[i], [W, H], use_udp=False)

    length=len(dis_16)
    # pdb.set_trace()
    ####
    # for idx in range(length):
    #     x=int(anchor_16[idx][0]*scale)
    #     y=int(anchor_16[idx][1]*scale)

    #     r = int(dis_16[idx]*scale)
    #     cv2.circle(img, (x, y), r, (0,128,255), 3 )
    #     cv2.circle(img, (x, y), 3, (255,0,0), -1)

    x=int(pt_16[0])
    y=int(pt_16[1])
    cv2.circle(img, (x, y), 3, (0,0,255), -1)

    return preds, img



from scipy.stats import entropy

def find_min_error_position(station,sample_points, sub_anchor_nk, Z):
    # Expand the dimension of sample_points to K*1*2
    sample_points_expanded = sample_points[:, np.newaxis]
    station=np.array(station)
    sub_anchor_nk=np.array(sub_anchor_nk)
    N=sub_anchor_nk.shape[0]
    sub_anchor_nk=sub_anchor_nk.reshape(1,N,2)
    station_expanded=station.reshape(1,1,-1)
    station_dis=station_expanded-sub_anchor_nk

    station_Dis=np.sum(station_dis**2,axis=-1)
    station_Dis=np.sqrt(station_Dis)

    sample_dis = sample_points_expanded - sub_anchor_nk
    sample_Dis = np.sum(sample_dis**2, axis=-1) 
    sample_Dis=np.sqrt(sample_Dis)

    Z=np.array(Z)
    E=abs(sample_Dis - Z)
    station_Dis=np.abs(station_Dis-Z) 
    station_error=np.sum(station_Dis,axis=-1)
    sum_error = np.sum(E, axis=-1)
    min_position = np.argmin(sum_error)

    return min_position,sum_error,station_error


def random_sample_around_rhomboid(heatmap, max_position, num_samples=10, radius=4):
    H, _ = heatmap.shape
    max_x, max_y = max_position
    H =int(H )
    sampling_distance = H / 128
    shape_size=8
    if H ==16:
        shape_size=4
    if H ==8:
        shape_size=3
    if H ==4:
        shape_size=2
    # shape_size =np.log2(H) #
    num_samples = int(shape_size / sampling_distance)

    sample_points_x, sample_points_y = np.meshgrid(
        np.linspace(max_x - (shape_size) / 2-sampling_distance/2,
                    max_x + (shape_size) / 2-sampling_distance/2, num_samples),
        np.linspace(max_y - (shape_size) / 2-sampling_distance/2,
                    max_y + (shape_size) / 2-sampling_distance/2, num_samples)
    )

    return np.column_stack((sample_points_x.ravel(), sample_points_y.ravel()))


def keypoints_from_heatmaps_randomsample_lls(heatmaps, preds, center, scale, sigma=2, phase=False):
    N, K, H, W = heatmaps.shape
    x = np.arange(0, W, 1, np.float32)
    y = np.arange(0, H, 1, np.float32)

    anchor=[]
    for i in range(0, H):
        for j in range(0, W):
            anchor.append([j, i])
    anchor = np.array(anchor)
    
    topk=30
    if H==32:
        topk=10
    elif H==16:
        topk=20
    elif H==8:
        topk=15
    elif H==4:
        topk=10
   
    sub_prob, sub_anchor = topK( heatmaps, anchor, topK=[0,topk] )
    max_position, maxvals = get_max_preds(heatmaps)


    for n in range(N):
        for k in range(K):

            heatmap = heatmaps[n][k]
            sub_prob_lls, sub_anchor_lls = block_search(heatmap, preds[n][k], H, W, step_h=2, step_w=2, gl=False )
            sub_dis_lls = decode_dis(sub_prob_lls, sigma=sigma)
            sub_anchor_lls = anchor_map(sub_anchor_lls, x, y)
            station = LLS_solve(sub_anchor_lls, sub_dis_lls)
            sample_points=random_sample_around_rhomboid(heatmaps[n][k], station, num_samples=500, radius=2)
            sub_anchor_nk = sub_anchor[n][k]
            sub_dis = decode_dis([sub_prob[n][k]], sigma=sigma)
            min_position,sum_error,station_error = find_min_error_position(station,sample_points, sub_anchor_nk, sub_dis)
            preds[n][k][0] = sample_points[min_position][0]
            preds[n][k][1] = sample_points[min_position][1]

    return preds





def keypoints_from_heatmaps_randomsample_lls_img_max(img,heatmaps, preds, center, scale, sigma=2, phase=False):
    N, K, H, W = heatmaps.shape
    x = np.arange(0, W, 1, np.float32)
    y = np.arange(0, H, 1, np.float32)

    anchor=[]
    for i in range(0, H):
        for j in range(0, W):
            anchor.append([j, i])
    anchor = np.array(anchor)
    
    topk=30
    if H==32:
        topk=30
    elif H==16:
        topk=10
    elif H==8:
        topk=15
    elif H==4:
        topk=10
    scale=256/H
    # sub_prob, sub_anchor = topK( heatmaps, anchor, topK=[0,topk] )
    sub_prob, sub_anchor = topK( heatmaps, anchor, topK=[0,topk] )
    # max_position, maxvals = get_max_preds(heatmaps)
    if phase:
        preds_old, maxvals_old = get_max_preds(heatmaps)
        preds,maxvals=get_max4_preds(heatmaps)
    station_error= [[[] for _ in range(K)] for _ in range(N)]

    sum_error= [[[] for _ in range(K)] for _ in range(N)]

    for n in range(N):
        for k in range(K):
            # sub_dis = decode_dis([sub_prob[n][k]], sigma=sigma)
            # max_position = np.argmax([sub_prob[n][k]])

            heatmap = heatmaps[n][k]
            sub_prob_lls, sub_anchor_lls = block_search(heatmap, preds_old[n][k], H, W, step_h=2, step_w=2, gl=False )
            sub_dis_lls = decode_dis(sub_prob_lls, sigma=sigma)
            sub_anchor_lls = anchor_map(sub_anchor_lls, x, y)
            station = LLS_solve(sub_anchor_lls, sub_dis_lls)
            # sub_dis = decode_dis([sub_prob[n][k]], sigma=sigma)
            # station=LLS_solve(sub_anchor[n][k],sub_dis)
            sample_points=random_sample_around_rhomboid(heatmaps[n][k], station, num_samples=500, radius=2)
            # min_position = find_min_error_position(station,sample_points, sub_anchor_lls, sub_dis_lls)

            sub_anchor_nk = sub_anchor[n][k]
           
            # sample_points=random_sample_around_rhomboid(heatmaps[n][k], station, num_samples=500, radius=2)
            # sample_points=random_sample_around_max(heatmaps[n][k], station, num_samples=500, radius=2)
            sub_dis = decode_dis([sub_prob[n][k]], sigma=sigma)
            min_position,sum_error[n][k],station_error[n][k] = find_min_error_position(station,sample_points, sub_anchor_nk, sub_dis)
            #  min_position = find_min_error_position(sample_points, sub_anchor_nk, sub_dis)
            # min_position = find_min_error_position_median(sample_points, sub_anchor_nk, sub_dis)

            preds_old[n][k][0] = sample_points[min_position][0]*scale
            preds_old[n][k][1] = sample_points[min_position][1]*scale
            
            # preds[n][k][0]=station[0]
            # preds[n][k][1]=station[1]

            # preds_old[n][k][0] =station[0]*scale
            # preds_old[n][k][1] =station[1]*scale

            # print("三点定位输出",station)
            # print("随机找点",sample_points[min_position]) 

    preds=preds*scale
    sample_points=sample_points*scale

    return preds,sample_points,img,sum_error,station_error,preds_old 


def _taylor(heatmap, coord):
    """Distribution aware coordinate decoding method.

    Note:
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmap (np.ndarray[H, W]): Heatmap of a particular joint type.
        coord (np.ndarray[2,]): Coordinates of the predicted keypoints.

    Returns:
        np.ndarray[2,]: Updated coordinates.
    """
    H, W = heatmap.shape[:2]
    px, py = int(coord[0]), int(coord[1])
    if 1 < px < W - 2 and 1 < py < H - 2:
        dx = 0.5 * (heatmap[py][px + 1] - heatmap[py][px - 1])
        dy = 0.5 * (heatmap[py + 1][px] - heatmap[py - 1][px])
        dxx = 0.25 * (
            heatmap[py][px + 2] - 2 * heatmap[py][px] + heatmap[py][px - 2])
        dxy = 0.25 * (
            heatmap[py + 1][px + 1] - heatmap[py - 1][px + 1] -
            heatmap[py + 1][px - 1] + heatmap[py - 1][px - 1])
        dyy = 0.25 * (
            heatmap[py + 2 * 1][px] - 2 * heatmap[py][px] +
            heatmap[py - 2 * 1][px])
        derivative = np.array([[dx], [dy]])
        hessian = np.array([[dxx, dxy], [dxy, dyy]])
        if dxx * dyy - dxy**2 != 0:
            hessianinv = np.linalg.inv(hessian)
            offset = -hessianinv @ derivative
            offset = np.squeeze(np.array(offset.T), axis=0)
            coord += offset
    return coord
def _gaussian_blur(heatmaps, kernel=11):
    """Modulate heatmap distribution with Gaussian.
     sigma = 0.3*((kernel_size-1)*0.5-1)+0.8
     sigma~=3 if k=17
     sigma=2 if k=11;
     sigma~=1.5 if k=7;
     sigma~=1 if k=3;

    Note:
        - batch_size: N
        - num_keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        np.ndarray ([N, K, H, W]): Modulated heatmap distribution.
    """
    assert kernel % 2 == 1

    border = (kernel - 1) // 2
    batch_size = heatmaps.shape[0]
    num_joints = heatmaps.shape[1]
    height = heatmaps.shape[2]
    width = heatmaps.shape[3]
    for i in range(batch_size):
        for j in range(num_joints):
            origin_max = np.max(heatmaps[i, j])
            dr = np.zeros((height + 2 * border, width + 2 * border),
                          dtype=np.float32)
            dr[border:-border, border:-border] = heatmaps[i, j].copy()
            dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
            heatmaps[i, j] = dr[border:-border, border:-border].copy()
            heatmaps[i, j] *= origin_max / np.max(heatmaps[i, j])
    return heatmaps


def keypoints_from_heatmaps_onehot_img_max(img, heatmaps, preds, center, scale, sigma=2, phase=False):

    N, K, H, W = heatmaps.shape
    scale=256/H
    # if post_process == 'unbiased':  # alleviate biased coordinate
        # apply Gaussian distribution modulation.
    heatmaps = np.log( np.maximum(_gaussian_blur(heatmaps, kernel=11), 1e-10))
    for n in range(N):
        for k in range(K):
            preds[n][k] = _taylor(heatmaps[n][k], preds[n][k])*scale
    # elif post_process is not None:
        # add +/-0.25 shift to the predicted locations for higher acc.
    # for n in range(N):
    #     for k in range(K):
    #         heatmap = heatmaps[n][k]
    #         px = int(preds[n][k][0])
    #         py = int(preds[n][k][1])
    #         if 1 < px < W - 1 and 1 < py < H - 1:
    #             diff = np.array([
    #                 heatmap[py][px + 1] - heatmap[py][px - 1],
    #                 heatmap[py + 1][px] - heatmap[py - 1][px]
    #             ])
    #             preds[n][k] += np.sign(diff) * .25
 



def keypoints_from_heatmaps_tril_img_max(img, heatmaps, preds, center, scale, sigma=2, phase=False):
    import pdb
    # pdb.set_trace()
    # preds_old=preds
    if phase:
        preds_old, maxvals_old = get_max_preds(heatmaps)
        preds,maxvals=get_max4_preds(heatmaps)
        
    
    N, K, H, W = heatmaps.shape
    x = np.arange(0, W, 1, np.float32)
    y = np.arange(0, H, 1, np.float32)

    anchor_16=[[0,0],[0,0],[0,0],[0,0]]
    dis_16=[0,0,0,0]
    pt_16=[0,0]
    scale=256/H
    # scale=32 s
    for n in range(N):
        for k in range(K):
            heatmap = heatmaps[n][k]
            sub_prob, sub_anchor = block_search(heatmap, preds_old[n][k], H, W, step_h=2, step_w=2, gl=False )
            sub_dis = decode_dis(sub_prob, sigma=sigma)
            sub_anchor = anchor_map(sub_anchor, x, y)
            station = LLS_solve(sub_anchor, sub_dis)

            # preds[n][k][0]=station[0]
            # preds[n][k][1]=station[1]
            preds_old[n][k][0]=station[0]*scale
            preds_old[n][k][1]=station[1]*scale

    return preds, img, preds_old
# # #-------------2023-04-23------------
# # #------------gauss_newton-iteration--------
def keypoints_from_heatmaps_tril_topK(heatmaps, preds, center, scale, sigma=2, phase=False):
    if phase:
        preds, maxvals = get_max_preds(heatmaps)
    
    N, K, H, W = heatmaps.shape
    x = np.arange(0, W, 1, np.float32)
    y = np.arange(0, H, 1, np.float32)

    anchor=[]
    for i in range(0, H):
        for j in range(0, W):
            anchor.append([j, i])
    anchor = np.array(anchor)
    
    topk=30
    if H==32:
        topk=30
    elif H==16:
        topk=30
    elif H==8:
        topk=15
    elif H==4:
        topk=10

    
    sub_prob, sub_anchor = topK( heatmaps, anchor, topK=[0,topk] )

    for n in range(N):
        for k in range(K):
            sub_dis = decode_dis([sub_prob[n][k]], sigma=sigma)
            station = fit_gauss_newton(preds[n][k], sub_anchor[n][k], sub_dis) #---------------
            
            preds[n][k][0]=station[0]
            preds[n][k][1]=station[1]

    return preds



