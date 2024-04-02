
import numpy as np

def robertedge(model, test_input, tar):
    prediction = model(test_input, training=True)
    ## calculate EDGE
    edge = []
    tar = tar[0].numpy()
    prediction = prediction[0].numpy()
    
    ## detect target 90 percentile
    for i in range(0,111):
        for j in range(0,37):
            for band in range(0,1):
                curr_edge = abs(tar[i][j][band]-tar[i+1][j+1][band])+abs(tar[i+1][j][band]-tar[i][j+1][band])
                edge.append(curr_edge)
                
    target_edge = np.percentile(edge, 90)
    
    edge_true = []
    for i in range(0,111):
        for j in range(0,37):
            for band in range(0,1):
                edge_prediction = abs(prediction[i][j][band]-prediction[i+1][j+1][band])+abs(prediction[i+1][j][band]-prediction[i][j+1][band])
                ## larger than 90 percentile
                if (edge_prediction>target_edge):
                    ## (sf-sr)/(sf+sr)
                    sf = edge_prediction
                    sr = abs(tar[i][j][band]-tar[i+1][j+1][band])+abs(tar[i+1][j][band]-tar[i][j+1][band])
                    edge_true.append((sf-sr)/(sf+sr))
    robertedge = np.mean(edge_true)
    
    return robertedge



def get_val(di, dc):
    if (di>dc):
        return 1
    return 0

def LBP(model, test_input, tar):
    prediction = model(test_input, training=True)
    prediction = prediction[0].numpy()
    target = tar[0].numpy()
    lbp = []
    for i in range(1,111):
        for j in range(1,37):
            for band in range(0,1):
                dc = prediction[i][j][band]
                d1 = get_val(prediction[i-1][j-1][band],dc)
                d2 = get_val(prediction[i-1][j][band],dc)
                d3 = get_val(prediction[i-1][j+1][band],dc)
                d4 = get_val(prediction[i][j-1][band],dc)
                d5 = get_val(prediction[i][j+1][band],dc)
                d6 = get_val(prediction[i+1][j-1][band],dc)
                d7 = get_val(prediction[i+1][j][band],dc)
                d8 = get_val(prediction[i+1][j+1][band],dc)
                
                dec_pred = d1*1+d2*2+d3*4+d4*8+d5*16+d6*32+d7*64+d8*128
                
                dc = target[i][j][band]
                d1 = get_val(target[i-1][j-1][band],dc)
                d2 = get_val(target[i-1][j][band],dc)
                d3 = get_val(target[i-1][j+1][band],dc)
                d4 = get_val(target[i][j-1][band],dc)
                d5 = get_val(target[i][j+1][band],dc)
                d6 = get_val(target[i+1][j-1][band],dc)
                d7 = get_val(target[i+1][j][band],dc)
                d8 = get_val(target[i+1][j+1][band],dc)
                
                dec_tar = d1*1+d2*2+d3*4+d4*8+d5*16+d6*32+d7*64+d8*128
                if ((dec_pred+dec_tar)!=0):
                    lbp.append((dec_pred-dec_tar)/(dec_pred+dec_tar))
    #return lbp
    return np.mean(lbp)