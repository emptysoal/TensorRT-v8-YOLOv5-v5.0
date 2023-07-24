#include "public.h"
#include "common.h"
#include "utils.h"
#include "preprocess.h"
#include "calibrator.h"

using namespace nvinfer1;


const std::string NET = "s";  // s / m / l / x
const char *      INPUT_NAME = "data";
const char *      OUTPUT_NAME = "prob";
const int         CLASS_NUM = Yolo::CLASS_NUM;
const int         INPUT_H = Yolo::INPUT_H;
const int         INPUT_W = Yolo::INPUT_W;
const int         OUTPUT_SIZE = 1 + Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float);  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1;
const float       NMS_THRESH = 0.4;
const float       CONF_THRESH = 0.5;
const std::string wtsFile = "./para.wts";
const std::string trtFile = "./model.plan";
const std::string testDataPath = "./images/";  // 用于推理
static Logger     gLogger(ILogger::Severity::kERROR);

// for FP16 mode
const bool        bFP16Mode = false;
// for INT8 mode
const bool        bINT8Mode = false;
const std::string cacheFile = "./int8.cache";
const std::string calibrationDataPath = "./coco_calib";  // 用于 int8 量化


static int get_width(int x, float gw, int divisor = 8){
    return int(ceil((x * gw) / divisor)) * divisor;
}

static int get_depth(int x, float gd) {
    if (x == 1) return 1;
    int r = round(x * gd);
    if (x * gd - int(x * gd) == 0.5 && (int(x * gd) % 2) == 0) {
        --r;
    }
    return std::max<int>(r, 1);
}


void buildNetwork(INetworkDefinition* network, IOptimizationProfile* profile, IBuilderConfig* config, std::map<std::string, Weights>& weightMap)
{
    float gd = 0.0f, gw = 0.0f;
    if (NET[0] == 's') {
        gd = 0.33;
        gw = 0.50;
    } else if (NET[0] == 'm') {
        gd = 0.67;
        gw = 0.75;
    } else if (NET[0] == 'l') {
        gd = 1.0;
        gw = 1.0;
    } else if (NET[0] == 'x') {
        gd = 1.33;
        gw = 1.25;
    }

    ITensor* inputTensor = network->addInput(INPUT_NAME, DataType::kFLOAT, Dims32 {4, {-1, 3, INPUT_H, INPUT_W}});
    profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMIN, Dims32 {4, {1, 3, INPUT_H, INPUT_W}});
    profile->setDimensions(inputTensor->getName(), OptProfileSelector::kOPT, Dims32 {4, {4, 3, INPUT_H, INPUT_W}});
    profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMAX, Dims32 {4, {8, 3, INPUT_H, INPUT_W}});
    config->addOptimizationProfile(profile);

    /* ------ yolov5 backbone------ */
    auto focus0 = focus(network, weightMap, *inputTensor, 3, get_width(64, gw), 3, "model.0");
    auto conv1 = convBlock(network, weightMap, *focus0->getOutput(0), get_width(128, gw), 3, 2, 1, "model.1");
    auto bottleneck_csp2 = C3(network, weightMap, *conv1->getOutput(0), get_width(128, gw), get_width(128, gw), get_depth(3, gd), true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *bottleneck_csp2->getOutput(0), get_width(256, gw), 3, 2, 1, "model.3");
    auto bottleneck_csp4 = C3(network, weightMap, *conv3->getOutput(0), get_width(256, gw), get_width(256, gw), get_depth(9, gd), true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), get_width(512, gw), 3, 2, 1, "model.5");
    auto bottleneck_csp6 = C3(network, weightMap, *conv5->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(9, gd), true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), get_width(1024, gw), 3, 2, 1, "model.7");
    auto spp8 = SPP(network, weightMap, *conv7->getOutput(0), get_width(1024, gw), get_width(1024, gw), 5, 9, 13, "model.8");

    std::cout << "Succeeded building backbone!" << std::endl;

    /* ------ yolov5 head ------ */
    auto bottleneck_csp9 = C3(network, weightMap, *spp8->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.9");
    auto conv10 = convBlock(network, weightMap, *bottleneck_csp9->getOutput(0), get_width(512, gw), 1, 1, 1, "model.10");

    auto upsample11 = network->addResize(*conv10->getOutput(0));
    assert(upsample11);
    upsample11->setResizeMode(ResizeMode::kNEAREST);
    Dims32 dim11{4, {1, bottleneck_csp6->getOutput(0)->getDimensions().d[1], bottleneck_csp6->getOutput(0)->getDimensions().d[2], bottleneck_csp6->getOutput(0)->getDimensions().d[3]}};
    upsample11->setOutputDimensions(dim11);

    ITensor* inputTensors12[] = { upsample11->getOutput(0), bottleneck_csp6->getOutput(0) };
    auto cat12 = network->addConcatenation(inputTensors12, 2);
    auto bottleneck_csp13 = C3(network, weightMap, *cat12->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.13");
    auto conv14 = convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), get_width(256, gw), 1, 1, 1, "model.14");

    auto upsample15 = network->addResize(*conv14->getOutput(0));
    assert(upsample15);
    upsample15->setResizeMode(ResizeMode::kNEAREST);
    Dims32 dim15{4, {1, bottleneck_csp4->getOutput(0)->getDimensions().d[1], bottleneck_csp4->getOutput(0)->getDimensions().d[2], bottleneck_csp4->getOutput(0)->getDimensions().d[3]}};
    upsample15->setOutputDimensions(dim15);

    ITensor* inputTensors16[] = { upsample15->getOutput(0), bottleneck_csp4->getOutput(0) };
    auto cat16 = network->addConcatenation(inputTensors16, 2);

    auto bottleneck_csp17 = C3(network, weightMap, *cat16->getOutput(0), get_width(512, gw), get_width(256, gw), get_depth(3, gd), false, 1, 0.5, "model.17");

    std::cout << "Succeeded building head!" << std::endl;

    /* ------ detect ------ */
    IConvolutionLayer* det0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);
    auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), get_width(256, gw), 3, 2, 1, "model.18");
    ITensor* inputTensors19[] = { conv18->getOutput(0), conv14->getOutput(0) };
    auto cat19 = network->addConcatenation(inputTensors19, 2);
    auto bottleneck_csp20 = C3(network, weightMap, *cat19->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.20");
    IConvolutionLayer* det1 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);
    auto conv21 = convBlock(network, weightMap, *bottleneck_csp20->getOutput(0), get_width(512, gw), 3, 2, 1, "model.21");
    ITensor* inputTensors22[] = { conv21->getOutput(0), conv10->getOutput(0) };
    auto cat22 = network->addConcatenation(inputTensors22, 2);
    auto bottleneck_csp23 = C3(network, weightMap, *cat22->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.23");
    IConvolutionLayer* det2 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);
    
    auto yolo = addYoloLayer(network, weightMap, "model.24", std::vector<IConvolutionLayer*>{det0, det1, det2});
    yolo->getOutput(0)->setName(OUTPUT_NAME);
    network->markOutput(*yolo->getOutput(0));

    std::cout << "Succeeded building total network!" << std::endl;
}


ICudaEngine* getEngine()
{
    ICudaEngine *engine = nullptr;

    if (access(trtFile.c_str(), F_OK) == 0)
    {
        std::ifstream engineFile(trtFile, std::ios::binary);
        long int      fsize = 0;

        engineFile.seekg(0, engineFile.end);
        fsize = engineFile.tellg();
        engineFile.seekg(0, engineFile.beg);
        std::vector<char> engineString(fsize);
        engineFile.read(engineString.data(), fsize);
        if (engineString.size() == 0) { std::cout << "Failed getting serialized engine!" << std::endl; return nullptr; }
        std::cout << "Succeeded getting serialized engine!" << std::endl;

        IRuntime *runtime {createInferRuntime(gLogger)};
        engine = runtime->deserializeCudaEngine(engineString.data(), fsize);
        if (engine == nullptr) { std::cout << "Failed loading engine!" << std::endl; return nullptr; }
        std::cout << "Succeeded loading engine!" << std::endl;
    }
    else
    {
        IBuilder *            builder     = createInferBuilder(gLogger);
        INetworkDefinition *  network     = builder->createNetworkV2(1U << int(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
        IOptimizationProfile* profile     = builder->createOptimizationProfile();
        IBuilderConfig *      config      = builder->createBuilderConfig();
        config->setMaxWorkspaceSize(1 << 30);
        IInt8Calibrator *     pCalibrator = nullptr;
        if (bFP16Mode)
        {
            config->setFlag(BuilderFlag::kFP16);
        }
        if (bINT8Mode)
        {
            config->setFlag(BuilderFlag::kINT8);
            int batchSize = 4;
            pCalibrator = new Int8EntropyCalibrator2(batchSize, INPUT_W, INPUT_H, calibrationDataPath.c_str(), cacheFile.c_str());
            config->setInt8Calibrator(pCalibrator);
        }
        // load .wts
        std::map<std::string, Weights> weightMap = loadWeights(wtsFile);

        buildNetwork(network, profile, config, weightMap);

        std::cout << "Building engine, please wait for a while..." << std::endl;
        IHostMemory* engineString = builder->buildSerializedNetwork(*network, *config);
        std::cout << "Succeeded building serialized engine!" << std::endl;

        // Release host memory
        for (auto& mem : weightMap)
        {
            free((void*)(mem.second.values));
        }

        IRuntime* runtime {createInferRuntime(gLogger)};
        engine = runtime->deserializeCudaEngine(engineString->data(), engineString->size());
        if (engine == nullptr) { std::cout << "Failed building engine!" << std::endl; return nullptr; }
        std::cout << "Succeeded building engine!" << std::endl;

        if (bINT8Mode && pCalibrator != nullptr)
        {
            delete pCalibrator;
        }

        std::ofstream engineFile(trtFile, std::ios::binary);
        engineFile.write(static_cast<char *>(engineString->data()), engineString->size());
        std::cout << "Succeeded saving .plan file!" << std::endl;
    }

    return engine;
}


void inference_one(IExecutionContext* context, float* inputData, float* outputData, std::vector<void *> vBufferD, std::vector<int> vTensorSize)
{
    CHECK(cudaMemcpy(vBufferD[0], (void *)inputData, vTensorSize[0], cudaMemcpyHostToDevice));

    context->executeV2(vBufferD.data());

    CHECK(cudaMemcpy((void *)outputData, vBufferD[1], vTensorSize[1], cudaMemcpyDeviceToHost));
}


int run()
{
    ICudaEngine* engine = getEngine();

    IExecutionContext* context = engine->createExecutionContext();
    context->setBindingDimensions(0, Dims32 {4, {1, 3, INPUT_H, INPUT_W}});

    std::vector<int> vTensorSize(2, 0);  // bytes of input and output
    for (int i = 0; i < 2; i++)
    {
        Dims32 dim = context->getBindingDimensions(i);
        int size = 1;
        for (int j = 0; j < dim.nbDims; j++)
        {
            size *= dim.d[j];
        }
        vTensorSize[i] = size * dataTypeToSize(engine->getBindingDataType(i));
    }

    // prepare input data and output data ---------------------------
    static float inputData[3 * INPUT_H * INPUT_W];
    static float outputData[OUTPUT_SIZE];
    //  prepare input and output space on device
    std::vector<void *> vBufferD (2, nullptr);
    for (int i = 0; i < 2; i++)
    {
        CHECK(cudaMalloc(&vBufferD[i], vTensorSize[i]));
    }

    // get image file names for inferencing
    std::vector<std::string> file_names;
    if (read_files_in_dir(testDataPath.c_str(), file_names) < 0) {
        std::cout << "read_files_in_dir failed." << std::endl;
        return -1;
    }
    // inference
    int total_cost = 0;
    int img_count = 0;
    for (int i = 0; i < file_names.size(); i++)
    {
        std::string testImagePath = testDataPath + "/" + file_names[i];
        cv::Mat img = cv::imread(testImagePath, cv::IMREAD_COLOR);
        if (img.empty()) continue;

        auto start = std::chrono::system_clock::now();

        preprocess(img, inputData, INPUT_H, INPUT_W);  // put image data on inputData

        inference_one(context, inputData, outputData, vBufferD, vTensorSize);

        std::vector<Yolo::Detection> res;
        nms(res, outputData, CONF_THRESH, NMS_THRESH);
        for (size_t j = 0; j < res.size(); j++)
        {
            cv::Rect r = get_rect(img, res[j].bbox);
            cv::rectangle(img, r, cv::Scalar(255, 0, 255), 2);
            cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(255, 255, 255), 2);
        }

        auto end = std::chrono::system_clock::now();

        cv::imwrite("_" + file_names[i], img);

        total_cost += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        img_count++;
    }

    int avg_cost = total_cost / img_count;
    std::cout << "Total image num is: " << img_count;
    std::cout << " inference total cost is: " << total_cost << "ms";
    std::cout << " average cost is: " << avg_cost << "ms" << std::endl;

    // free device memory
    for (int i = 0; i < 2; ++i)
    {
        CHECK(cudaFree(vBufferD[i]));
    }

    return 0;
}

int main()
{
    CHECK(cudaSetDevice(0));
    run();
    return 0;
}
