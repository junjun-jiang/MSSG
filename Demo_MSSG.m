clear all; close all; clc;warning('off'),
addpath(genpath(cd));
addpath('.\HyperData');
datasets = {'Indian','PaviaU','Salinas'};
% datasets = {'Indian'};
for idataset = 1:length(datasets)
    dataset = datasets{idataset};

    if strcmp(dataset,'Indian')==1
        load Indian_pines_corrected;load Indian_pines_gt;load Indian_pines_randp %s=2 10^1 0.01
        paviaU = indian_pines_corrected;
        paviaU_gt = indian_pines_gt;
        trainnumber = 0.1; Ratio = 0.0812;%Ratio = N_f/N_I;    % the value of Ratio can be obtained by the function of "Edge_ratio3" in the function of "cubseg"
    elseif strcmp(dataset,'PaviaU')==1
        load PaviaU;load PaviaU_gt;load PaviaU_randp 
        trainnumber = 50; Ratio = 0.0664;%Ratio = N_f/N_I;    
    elseif strcmp(dataset,'Salinas')==1
        load Salinas_corrected;load Salinas_gt;load Salinas_randp %  s=2 10^1 0.01
        paviaU = salinas_corrected;
        paviaU_gt = salinas_gt;
        trainnumber = 50; Ratio = 0.0513;%Ratio = N_f/N_I;    
    end

    % smoothing filter to the HSI
    for i=1:size(paviaU,3);
        paviaU(:,:,i) = imfilter(paviaU(:,:,i),fspecial('average',3));
    end

    % multi-scale superpixel segmentation, we obtain 2*C+1 layer segmentations
    labelms = [];
    C = 10;
    T_base = 2000;alpha   = sqrt(2);
    str_mseg = [dataset ' multiscale seg S=',num2str(T_base),'_A=',num2str(alpha),'_C=',num2str(C),'.mat'];
    if exist(str_mseg,'file')
        load(str_mseg,'labelms');
    else
        fprintf('\n Multiscale superpixel segmentation.\n');
        segs = T_base*Ratio*alpha.^[-C:C];
        for iseg = 1:size(segs,2)
            labelms(:,:,iseg) = cubseg(paviaU,segs(iseg));
        end
        save(str_mseg,'labelms');
        fprintf('Done.\n');
    end

    L1 = 4; % L1 in Eq.(10)
    labels = labelms(:,:,C+1-L1:C+1+2*L1);

%     rhos   =[0.1:0.1:0.5];
    rhos   =[0.3];

%     classifier = {'NN','SVM','RF','ELM'};
    classifier = {'ELM'};

    for iclassifier = 1:length(classifier)

        switch classifier{iclassifier}
            case 'NN'
                alphaa   = [1]; % NN
            case 'ELM'                        
                alphaa   = 2.^[6 8 10 12]; % elm
            case 'SVM'
                alphaa   = [0.01 0.1 1 10 100 1000]; % SVM
            case 'RF'
                alphaa   = [100];
        end

        for iter = 1:10 % run the method by 10 times to avoid randomness

            randpp=randp{iter};  
            % randomly divide the dataset to training and test samples    
            [DataTest, DataTrain, CTest, CTrain, Loc_test] = samplesdivide(paviaU,paviaU_gt,trainnumber,randpp);

            for irho = 1:size(rhos,2) 
        
                % multilayer spectral-spatial graphs geneartion and fusion
                [A] = supSimALLMultiscale(paviaU,paviaU_gt,trainnumber,randpp,labels,size(DataTrain,1));  

                fprintf('\nDataset:%7s, Classifier: %2s, Round: %2d, Noise ratio (rho): %.4f\n', dataset, classifier{iclassifier}, iter, rhos(irho));
                % Get label from the class num
                trainlabel = getlabel(CTrain);  
                testlabel  = getlabel(CTest);  

                % add noise to the label
                trainlabel_nl = label2noisylabel(trainlabel,rhos(irho));       
                fprintf('Noisy pixel number =%3d ',length(find(trainlabel_nl-trainlabel~=0)));

                % noisy label cleansing with A
                [trainlabel_cl] = labelpropagation(trainlabel_nl, A);                 
                fprintf('and Noisy pixel number after cleansing = %3d\n',length(find(trainlabel_cl-trainlabel~=0)));
                DataTrain_nl = DataTrain;  
                
                tempaccy = 0;     
                tempconfusion = [];            
                for ialpha = 1:size(alphaa,2)
                    alpha = alphaa(ialpha); % alpha is the parameter of different classifiers
                    switch classifier{iclassifier}
                        case 'NN'
                            Predict_label = knnclassify(DataTest,DataTrain_nl,trainlabel_cl,alpha,'euclidean');
                        case 'ELM'                        
                            [TTrain,TTest,TrainAC,accur_ELM,TY,Predict_label] = elm_kernel([trainlabel_cl' DataTrain_nl],[testlabel' DataTest],1,alpha,'RBF_kernel',1);
                        case 'SVM'
                            [Predict_label, ~,~] = svmpredict(testlabel', DataTest, svmtrain(trainlabel_cl', DataTrain_nl, ['-q -c 100000 -g ' num2str(alpha) ' -b 1']), '-b 1');    Predict_label =Predict_label';      
                        case 'RF'
                            Factor = TreeBagger(alpha, DataTrain_nl, trainlabel_cl);
                            [Predict_label_temp,Scores] = predict(Factor, DataTest);                        
                            for ij=1:length(Predict_label_temp); Predict_label(ij) = str2num(Predict_label_temp{ij}); end;     
                    end       
                    [tempconfusion, accur_NRS, TPR, FPR] = confusion_matrix_wei(Predict_label, CTest);                

                    if tempaccy<accur_NRS    
                        tempaccy = accur_NRS;
                        Predict_label_best = Predict_label;
                        confusion = tempconfusion;
                    end
                    tempaccys(ialpha) = accur_NRS;
                end  
                MSSG_accy(idataset,iclassifier,irho,iter) = max(tempaccys);     
                %MSSG_aas(idataset,iclassifier,irho,iter,:) = diag(confusion)./CTest';  
                MSSG_aa(idataset,iclassifier,irho,iter) = mean(diag(confusion)./CTest');  
                MSSG_ka(idataset,iclassifier,irho,iter) = cal_kappa(confusion);

                fprintf('MSSG: %.6f\n', max(tempaccys));      
                % put the prediect label of the testing samples back to the map to get the Classification Map (label_map)    
                label_map = class2label(Predict_label_best,trainlabel_nl,randpp,paviaU_gt,CTest,CTrain);  

                % save the classification map
                imwrite(label2color2(label_map,'colormap','jet'),['.\Classification maps\',dataset,'_',classifier{iclassifier},'_',num2str(iter),'_',num2str(rhos(irho)),'_MSSG.png']); 
           end
        end
    end
end
