
/*
 * Company:	Synthesis
 * Author: 	Li
 * Date:	2019.12.25	
 */

#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <cassert>

#include <string>
#include <vector>
#include <sys/time.h>
#include <glog/logging.h>

#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>

#include "image.h"
#include "gaussian_yolo_layer.h"
#include <unistd.h>
#include <dirent.h>
using namespace caffe;
using namespace cv;
//using namespace std;

#define max(a,b) (((a) > (b)) ? (a) : (b))
#define min(a,b) (((a) < (b)) ? (a) : (b))
bool signal_recieved = false;

void sig_handler(int signo){
    if( signo == SIGINT ){
            printf("received SIGINT\n");
            signal_recieved = true;
    }
}

uint64_t current_timestamp() {
    struct timeval te; 
    gettimeofday(&te, NULL); // get current time
    return te.tv_sec*1000LL + te.tv_usec/1000; // caculate milliseconds
}

vector<string> getFiles(string cate_dir)
{
	vector<string> files;//存放文件名
 
	DIR *dir;
	struct dirent *ptr;
	char base[1000];
 
	if ((dir=opendir(cate_dir.c_str())) == NULL)
        {
		perror("Open dir error...");
                exit(1);
        }
	while ((ptr=readdir(dir)) != NULL)
	{
		if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)    ///current dir OR parrent dir
		        continue;
		else if(ptr->d_type == 8)    ///file
			//printf("d_name:%s/%s\n",basePath,ptr->d_name);
			files.push_back(ptr->d_name);
		else if(ptr->d_type == 10)    ///link file
			//printf("d_name:%s/%s\n",basePath,ptr->d_name);
			continue;
		else if(ptr->d_type == 4)    ///dir
		{
			files.push_back(ptr->d_name);
			/*
		        memset(base,'\0',sizeof(base));
		        strcpy(base,basePath);
		        strcat(base,"/");
		        strcat(base,ptr->d_nSame);
		        readFileList(base);
			*/
		}
	}
	closedir(dir);
 
	//排序，按从小到大排序
	sort(files.begin(), files.end());
	return files;
}
float IOU(box box1,box box2){
    box iou_box;
    iou_box.x = max(box1.x,box2.x);
    iou_box.y = max(box1.y,box2.y);
    iou_box.w = min(box1.x+box1.w,box2.x+box2.w)-iou_box.x;
    iou_box.h = min(box1.y+box1.h,box2.y+box2.h)-iou_box.y;
    float area_iou = iou_box.w*iou_box.h*1.0;
    float area_sum = box1.w*box1.h*1.0+box2.w*box2.h*1.0;
    if (iou_box.w<0||iou_box.h<0){
        area_iou=0.0;
    }
    float iou =area_iou/(area_sum-area_iou);
    if (area_sum==0){
        return 0;
    }
    return iou;
}

vector<vector<int>> readann(string ann_path){
    string file=ann_path;
    //LOG(INFO) <<file;
    vector<vector<int>> gt_box;
    std::ifstream infile; 
    infile.open(file.data());   //将文件流对象与文件连接起来 
    if (!infile.is_open())
        {
         LOG(ERROR) <<"can't open ann";
        exit(0);
    }
    string s;
    while(getline(infile,s)){
        vector<int> temp;
        stringstream ss(s);
        string res=" ", tmp;  
        while (ss>>tmp){     //ss>>tmp，从字符串流读出一个字符串到tmp中，tmp遇到空格停止。比如输入 "xiao  yan",此时tmp中为"xiao",第二次循环读出时为"yan".
            if(tmp=="person") tmp="1";
            else if(tmp=="car") tmp="0";
            else if(tmp=="bicycle") tmp="2";
            temp.push_back(atoi(tmp.c_str()));
        }
        gt_box.push_back(temp);
    }
    return gt_box;
}

int main( int argc, char** argv )
{
    string model_file;
    string weights_file;
    string image_path;
    string ann_path;
    string save_path;
    if(6 == argc){
        model_file = argv[1];
        weights_file = argv[2];
        image_path = argv[3];
        ann_path = argv[4];
        save_path = argv[5];
    }
    else{
        LOG(ERROR) << "Input error: please input ./xx [model_path] [weights_path] [image_path] [ann_path] [save_path]";
        return -1;
    }	
    vector<float> sum_(3);
    vector<float> arr_(3);
    // Initialize the network.
    Caffe::set_mode(Caffe::GPU);

    image im,sized;
    vector<Blob<float>*> blobs;
    blobs.clear();

    int nboxes = 0;
    int size;
    detection *dets = NULL;
        
    /* load and init network. */
    shared_ptr<Net<float> > net;
    net.reset(new Net<float>(model_file, TEST));
    net->CopyTrainedLayersFrom(weights_file);
    LOG(INFO) << "net inputs numbers is " << net->num_inputs();
    LOG(INFO) << "net outputs numbers is " << net->num_outputs();

    CHECK_EQ(net->num_inputs(), 1) << "Network should have exactly one input.";

    Blob<float> *net_input_data_blobs = net->input_blobs()[0];
    LOG(INFO) << "input data layer channels is  " << net_input_data_blobs->channels();
    LOG(INFO) << "input data layer width is  " << net_input_data_blobs->width();
    LOG(INFO) << "input data layer height is  " << net_input_data_blobs->height();

    size = net_input_data_blobs->channels()*net_input_data_blobs->width()*net_input_data_blobs->height();
    
    //load image
    vector<string> file_names=getFiles(image_path);
    for(int u=0;u<file_names.size();u++){
        
        uint64_t beginDataTime =  current_timestamp();
        string image_name=file_names[u];
        //获得ground truth 
        string ann_p=ann_path+image_name.substr(0, image_name.length() - 4)+".txt";
        LOG(INFO) <<ann_p;
        vector<vector<int>> gt_box=readann(ann_p);
        
        
        string image_p=image_path+image_name;
        LOG(INFO) <<image_p;
        im = load_image_color((char*)image_p.c_str(),0,0);
        sized = letterbox_image(im,net_input_data_blobs->width(),net_input_data_blobs->height());
        cuda_push_array(net_input_data_blobs->mutable_gpu_data(),sized.data,size);

        uint64_t endDataTime =  current_timestamp();
        LOG(INFO) << "processing data operation avergae time is "
                << endDataTime - beginDataTime << " ms";

        uint64_t startDetectTime = current_timestamp();
        // forward
        net->Forward();
        for(int i=0;i<net->num_outputs();++i){
            blobs.push_back(net->output_blobs()[i]);
        }
        //LOG(INFO) << blobs.size();
        dets = get_detections(blobs,im.w,im.h,
            net_input_data_blobs->width(),net_input_data_blobs->height(),&nboxes);

        uint64_t endDetectTime = current_timestamp();
        LOG(INFO) << "caffe yolov3 : processing network yolov3 tiny avergae time is "
                << endDetectTime - startDetectTime << " ms";
    
        //show detection results
        Mat img = imread(image_p.c_str());
        int i,j;
        vector<vector<int>> pre_box;
        for(i=0;i< nboxes;++i){
            char labelstr[4096] = {0};
            int cls = -1;
            for(j=0;j<3;++j){
                if(dets[i].prob[j] > 0.5){
                    if(cls < 0){
                        cls = j;
                    }
                    LOG(INFO) << "label = " << cls
                            << ", prob = " << dets[i].prob[j]*100;
                }
            }
            vector<int> t;
            if(cls >= 0){
                
                t.push_back(cls);
                box b = dets[i].bbox;
                int left  = (b.x-b.w/2.)*im.w;
                int right = (b.x+b.w/2.)*im.w;
                int top   = (b.y-b.h/2.)*im.h;
                int bot   = (b.y+b.h/2.)*im.h;
                if(cls==0) rectangle(img,Point(left,top),Point(right,bot),Scalar(255,0,0),3,8,0);
                else if(cls==1) rectangle(img,Point(left,top),Point(right,bot),Scalar(0,0,255),3,8,0);
                else if(cls==2) rectangle(img,Point(left,top),Point(right,bot),Scalar(0,255,0),3,8,0);
                t.push_back(left);
                t.push_back(top);
                t.push_back(right);
                t.push_back(bot);
                LOG(INFO) << " left = " << left
                        << ", right = " << right
                        << ", top = " << top
                        << ", bot = " << bot;
                pre_box.push_back(t);
            }
            
        }

        //计算map:gt_box,pre_box;
        box bbox_gt;
        for (int i=0;i<gt_box.size();i++){
            bbox_gt.x=gt_box[i][1];
            bbox_gt.y=gt_box[i][2];
            bbox_gt.w=gt_box[i][3]-gt_box[i][1];
            bbox_gt.h=gt_box[i][4]-gt_box[i][2];
            //std::cout<<gt_box[i][0]-'0'<<std::endl;
            sum_[gt_box[i][0]]++;
            box bbox_pre;
            for (int j=0;j<pre_box.size();j++){

                bbox_pre.x=pre_box[j][1];
                bbox_pre.y=pre_box[j][2];
                bbox_pre.w=pre_box[j][3]-pre_box[j][1];
                bbox_pre.h=pre_box[j][4]-pre_box[j][2];
                //LOG(INFO) <<"gt cls:"<< gt_box[i][0]<<",pre cls:"<<pre_box[j][0]<<",IOU:"<<IOU(bbox_gt,bbox_pre);
                //LOG(INFO) <<"gt box:"<< bbox_gt.x<<","<<bbox_gt.y<<","<<bbox_gt.w<<","<<bbox_gt.h;
                //LOG(INFO) <<"pre box:"<< bbox_pre.x<<","<<bbox_pre.y<<","<<bbox_pre.w<<","<<bbox_pre.h;
                if((IOU(bbox_gt,bbox_pre)>0.5)&&((gt_box[i][0])==pre_box[j][0])){
                    arr_[pre_box[j][0]]++;
                    break;
                }
            }
        }
        
        //namedWindow("show",CV_WINDOW_AUTOSIZE);
        //imshow("show",img);
        LOG(INFO) <<save_path+image_name;
        LOG(INFO) <<imwrite(save_path.c_str()+image_name,img);
        imwrite(save_path.c_str()+image_name,img);
        //waitKey(0);

        free_detections(dets,nboxes);
        free_image(im);
        free_image(sized);
        blobs.clear();
    }  
    LOG(INFO) <<"car arr"<<arr_[0]<<",car sum"<<sum_[0]<<",mAP:"<<arr_[0]/sum_[0];
    LOG(INFO) <<"person arr"<<arr_[1]<<",person sum"<<sum_[1]<<",mAP:"<<arr_[1]/sum_[1];
    LOG(INFO) <<"bicycle arr"<<arr_[2]<<",bicycle sum"<<sum_[2]<<",mAP:"<<arr_[2]/sum_[2];
    LOG(INFO) << "done.";
    return 0;
}

