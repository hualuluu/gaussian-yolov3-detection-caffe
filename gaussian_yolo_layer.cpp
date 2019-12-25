/*
 * Company:	Synthesis
 * Author: 	Li
 * Date:	2019.12.25
 */

#include "gaussian_yolo_layer.h"
#include "blas.h"
#include "cuda.h"
#include "activations.h"
#include "box.h"
#include <stdio.h>
#include <math.h>

//yolov3
float biases[18] = {7,10, 14,24, 27,43, 32,97, 57,64, 92,109, 73,175, 141,178, 144,291};

//yolov3-tiny
float biases_tiny[12] = {10,14,23,27,37,58,81,82,135,169,344,319};

layer make_gaussian_yolo_layer(int batch,int w,int h,int net_w,int net_h,int n,int total,int classes)
{
    layer l = {0};
    l.n = n;
    l.total = total;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = n*(classes+ 8 + 1);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.inputs = l.w*l.h*l.c;

    l.biases = (float*)calloc(total*2,sizeof(float));

    l.mask = (int*)calloc(n,sizeof(int));
    if(9 == total){
        for(int i =0;i<total*2;++i){
            l.biases[i] = biases[i];
        }
        if(l.w == net_w / 32){
            int j = 6;
            for(int i =0;i<l.n;++i)
                l.mask[i] = j++;
        }
        if(l.w == net_w / 16){
            int j = 3;
            for(int i =0;i<l.n;++i)
                l.mask[i] = j++;
        }
        if(l.w == net_w / 8){
            int j = 0;
            for(int i =0;i<l.n;++i)
                l.mask[i] = j++;
        }
    }

    if(6 == total){
        for(int i =0;i<total*2;++i){
            l.biases[i] = biases_tiny[i];
        }
        if(l.w == net_w / 32){
            int j = 3;
            for(int i =0;i<l.n;++i)
                l.mask[i] = j++;
        }
        if(l.w == net_w / 16){
            int j = 0;
            for(int i =0;i<l.n;++i)
                l.mask[i] = j++;
        }
    }
    l.outputs = l.inputs;
    l.output = (float*)calloc(batch*l.outputs,sizeof(float));
    l.output_gpu = cuda_make_array(l.output,batch*l.outputs);
    
    return l;
}

void free_gaussian_yolo_layer(layer l)
{
    if(NULL != l.biases){
        free(l.biases);
        l.biases = NULL;
    }

    if(NULL != l.mask){
        free(l.mask);
        l.mask = NULL;
    }
    if(NULL != l.output){
        free(l.output);
        l.output = NULL;
    }

    if(NULL != l.output_gpu)
        cuda_free(l.output_gpu);
}

static int entry_gaussian_index(layer l, int batch, int location, int entry)
{
    int n =   location / (l.w*l.h);
    //LOG(INFO)<<n;
    int loc = location % (l.w*l.h);
    //LOG(INFO)<<loc;
    return batch*l.outputs + n*l.w*l.h*(8+l.classes+1) + entry*l.w*l.h + loc;
}

void forward_gaussian_yolo_layer_gpu(const float* input,layer l)
{
    copy_gpu(l.batch*l.inputs,(float*)input,1,l.output_gpu,1);
    int b,n;
    for(b = 0;b < l.batch;++b){
        for(n =0;n< l.n;++n){
            // x : mu, sigma
            int index = entry_gaussian_index(l, b, n*l.w*l.h, 0);
            activate_array_gpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC);
            // y : mu, sigma
            index = entry_gaussian_index(l, b, n*l.w*l.h, 2);
            activate_array_gpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC);
            // w : sigma
            index = entry_gaussian_index(l, b, n*l.w*l.h, 5);
            activate_array_gpu(l.output_gpu + index, l.w*l.h, LOGISTIC);
            // h : sigma
            index = entry_gaussian_index(l, b, n*l.w*l.h, 7);
            activate_array_gpu(l.output_gpu + index, l.w*l.h, LOGISTIC);
            // objectness & class
            index = entry_gaussian_index(l, b, n*l.w*l.h, 8);
            activate_array_gpu(l.output_gpu + index, (1+l.classes)*l.w*l.h, LOGISTIC);
        }
    }
    cuda_pull_array(l.output_gpu,l.output,l.batch*l.outputs);
}



int gaussian_yolo_num_detections(layer l,float thresh)//筛选出yolo生成的anchorz中满足阈值的框，返回满足的框的数目
{
    int i,n,b;
    int count = 0;
    for(b = 0;b < l.batch;++b){
        for(i=0;i<l.w*l.h;++i){
            for(n=0;n<l.n;++n){
                int obj_index = entry_gaussian_index(l,b,n*l.w*l.h+i,8);
                if(l.output[obj_index] > thresh)
                    ++count;
            }
        }
    }
    return count;
}

int gaussian_num_detections(vector<layer> layers_params,float thresh)//使用yolo_num_detection统计所有特征layer上满足阈值的anchor数目(3个feature map)
{
    int i;
    int s=0;
    for(i=0;i<layers_params.size();++i){
        layer l  = layers_params[i];
        s += gaussian_yolo_num_detections(l,thresh);
    }
    return s;

}

detection* make_network_boxes(vector<layer> layers_params,float thresh,int* num)//创造box的存储空间(坐标和概率)
{
    layer l = layers_params[0];
    int i;
    int nboxes = gaussian_num_detections(layers_params,thresh);//统计所有尺度下满足阈值的box的数目
    
    if(num) *num = nboxes;
    detection *dets = (detection*)calloc(nboxes,sizeof(detection));//给检测box坐标腾出空间
    
    for(i=0;i<nboxes;++i){
        dets[i].prob = (float*)calloc(l.classes,sizeof(float));//给检测box的概率腾出空间
        dets[i].uc = (float*)calloc(4,sizeof(float));
        //if(l.coords > 4)
        //{
        //    dets[i].mask = (float*)(l.coords-4,sizeof(float));
        //}
    }
    return dets;
}


void correct_gaussian_yolo_boxes(detection* dets,int n,int w,int h,int netw,int neth,int relative)
{//正确的yolo box--将坐标值转换为真实的框值
	// resize 以长边为先，短边填充
    // 此处new_w表示输入图片经压缩后在网络输入大小的letter_box中的width,new_h表示在letter_box中的height,
	// 以1280*720的输入图片为例，在进行letter_box的过程中，原图经resize后的width为416， 那么resize后的对应height为720*416/1280,
	//所以height为234，而超过234的上下空余部分在作为网络输入之前填充了128，new_h=234
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h)){
        // 如果w>h说明resize的时候是以width/图像的width为resize比例的，先得到中间图的width,再根据比例得到height
        new_w = netw;
        new_h = (h * netw)/w;
    }
    else{
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (i = 0; i < n; ++i){
        box b = dets[i].bbox;
        // 此处的公式很不好理解还是接着上面的例子，现有new_w=416,new_h=234,因为resize是以w为长边压缩的
		// 所以x相对于width的比例不变，而b.y表示y相对于图像高度的比例，在进行这一步的转化之前，b.y表示
		// 的是预测框的y坐标相对于网络height的比值，要转化到相对于letter_box中图像的height的比值时，需要先
		// 计算出y在letter_box中的相对坐标，即(b.y - (neth - new_h)/2./neth)，再除以比例
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw);
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth);
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}


box get_gaussian_yolo_box(float* x,float* biases,int n,int index,int i,int j,int lw, int lh,int w, int h,int stride)
{//坐标变换公式
    box b;
    b.x = (i + x[index + 0*stride]) / lw;
    b.y = (j + x[index + 2*stride]) / lh;
    b.w = exp(x[index + 4*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 6*stride]) * biases[2*n+1] / h;
    return b;
}


int get_gaussian_yolo_detections(layer l,int w, int h, int netw,int neth,float thresh,int *map,int relative,detection *dets)
{//获取yolo layer生成的box，返回的是存放box的数目
    int i,j,n,b;
    float* predictions = l.output;
    int count = 0;
    for(b = 0;b < l.batch;++b){//对每个feature尺度的每个batch循环
    for(i=0;i<l.w*l.h;++i){
        int row = i/l.w;
        int col = i%l.w;
        for(n = 0;n<l.n;++n){    
            int obj_index  = entry_gaussian_index(l, 0, n*l.w*l.h + i, 8);
            float objectness = predictions[obj_index];
            if(objectness <= thresh) continue;
            int box_index  = entry_gaussian_index(l, 0, n*l.w*l.h + i, 0);
            dets[count].bbox = get_gaussian_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h);
            
            dets[count].objectness = objectness;
            dets[count].classes = l.classes;
            dets[count].uc[0]=0.0;
             dets[count].uc[0] = predictions[entry_gaussian_index(l, 0, n*l.w*l.h + i, 1)]; // tx uncertainty
            dets[count].uc[1] = predictions[entry_gaussian_index(l, 0, n*l.w*l.h + i, 3)]; // ty uncertainty
            dets[count].uc[2] = predictions[entry_gaussian_index(l, 0, n*l.w*l.h + i, 5)]; // tw uncertainty
            dets[count].uc[3] = predictions[entry_gaussian_index(l, 0, n*l.w*l.h + i, 7)]; // th uncertainty
            for(j = 0; j < l.classes; ++j){
                int class_index = entry_gaussian_index(l, 0, n*l.w*l.h + i, 9 + j);
                float uc_aver = (dets[count].uc[0] + dets[count].uc[1] + dets[count].uc[2] + dets[count].uc[3])/4.0;
                float prob = objectness*predictions[class_index]*(1.0-uc_aver);
                dets[count].prob[j] = (prob > thresh) ? prob : 0;
            }
            ++count;
        }
    }
  }
    correct_gaussian_yolo_boxes(dets,count,w,h,netw,neth,relative);
    return count;
}


void fill_network_boxes(vector<layer> layers_params,int img_w,int img_h,int net_w,int net_h,float thresh, float hier, int *map,int relative,detection *dets)
{   //按数量构建好所有可能框的坐标和概率空间以后，就利用本函数去填充这些空间--将有效的box进行存储
    int j;
    for(j=0;j<layers_params.size();++j){
        layer l = layers_params[j];
        int count = get_gaussian_yolo_detections(l,img_w,img_h,net_w,net_h,thresh,map,relative,dets);
        dets += count;
    }
}


detection* get_network_boxes(vector<layer> layers_params,
                             int img_w,int img_h,int net_w,int net_h,float thresh,float hier,int* map,int relative,int *num)
{
    //make network boxes
    detection *dets = make_network_boxes(layers_params,thresh,num);
    //fill network boxes
    fill_network_boxes(layers_params,img_w,img_h,net_w,net_h,thresh,hier,map,relative,dets);
    
    return dets;
}

//get detection result
detection* get_detections(vector<Blob<float>*> blobs,int img_w,int img_h,int net_w,int net_h,int *nboxes)
{   
    vector<layer> layers_params;
    layers_params.clear();
    
    for(int i=0;i<blobs.size();++i){
        layer l_params;
        //network init
        l_params = make_gaussian_yolo_layer(blobs[i]->num(),blobs[i]->width(),blobs[i]->height(),net_w,net_h,num_bboxes,blobs.size()*dev_num_anchors,classes);
       
        layers_params.push_back(l_params);
        
        forward_gaussian_yolo_layer_gpu(blobs[i]->gpu_data(),l_params);
    }
    //get network boxes获得所有满足阈值的box(先构建空间，再填充内容)
    detection* dets = get_network_boxes(layers_params,img_w,img_h,net_w,net_h,thresh,hier_thresh,0,relative,nboxes);
    
    //release layer memory
    for(int index =0;index < layers_params.size();++index){
        free_gaussian_yolo_layer(layers_params[index]);
    }

    //do nms 做非极大值抑制
    if(nms_thresh) do_nms_sort(dets,(*nboxes),classes,nms_thresh);

    return dets;       
}


//release detection memory
void free_detections(detection *dets,int nboxes)
{
    int i;
    for(i = 0;i<nboxes;++i){
        free(dets[i].prob);
    }
    free(dets);
}
