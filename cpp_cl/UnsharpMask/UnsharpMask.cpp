/**********************************************************************
Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

•   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
•   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#include "UnsharpMask.hpp"

inline Pixel
UnsharpMask::getPixel(unsigned char* image, int x, int y, int width)
{
    Pixel p;
    unsigned char* ptr = image + (width*y+x)*4;
    p.b = *ptr;
    p.g = *(ptr+1);
    p.r = *(ptr+2);
    p.a = *(ptr+3);
    return p;
}

inline void
UnsharpMask::setPixel(unsigned char* image, int x, int y, int width, Pixel p)
{
    unsigned char* ptr = image + (width*y+x)*4;
    *ptr = p.b;
    *(ptr+1) = p.g;
    *(ptr+2) = p.r;
    *(ptr+3) = p.a;
}

int
UnsharpMask::initialize()
{
    // Call base class Initialize to get default configuration
    CHECK_ERROR(sampleArgs->initialize(), SDK_SUCCESS,
                "OpenCL resource  initialization failed");

    Option* radius_option = new Option;
    CHECK_ALLOCATION(radius_option, "Memory Allocation error.\n");

    radius_option->_sVersion = "r";
    radius_option->_lVersion = "radius";
    radius_option->_description = "Specify the Radius";
    radius_option->_type     = CA_ARG_INT;
    radius_option->_usage = "[value]";
    radius_option->_value    = &radius;

    sampleArgs->AddOption(radius_option);
    delete radius_option;

    Option* threshold_option = new Option;
    CHECK_ALLOCATION(threshold_option, "Memory Allocation error.\n");

    threshold_option->_sVersion = "";
    threshold_option->_lVersion = "threshold";
    threshold_option->_description = "Specify the Threshold";
    threshold_option->_type     = CA_ARG_FLOAT;
    threshold_option->_usage = "[value]";
    threshold_option->_value    = &threshold;

    sampleArgs->AddOption(threshold_option);
    delete threshold_option;

    Option* amount_option = new Option;
    CHECK_ALLOCATION(amount_option, "Memory Allocation error.\n");

    amount_option->_sVersion = "a";
    amount_option->_lVersion = "amount";
    amount_option->_description = "Specify the Amount in %";
    amount_option->_type     = CA_ARG_FLOAT;
    amount_option->_usage = "[value]";
    amount_option->_value    = &amount;
    sampleArgs->AddOption(amount_option);
    delete amount_option;

    Option* gui_option = new Option;
    CHECK_ALLOCATION(gui_option, "Memory Allocation error.\n");

    gui_option->_sVersion = "g";
    gui_option->_lVersion = "gui";
    gui_option->_description = "Run GUI";
    gui_option->_type     = CA_NO_ARGUMENT;
    gui_option->_value    = &gui;

    sampleArgs->AddOption(gui_option);
    delete gui_option;


    Option* iteration_option = new Option;
    CHECK_ALLOCATION(iteration_option, "Memory allocation error.\n");

    iteration_option->_sVersion = "i";
    iteration_option->_lVersion = "iterations";
    iteration_option->_description = "Specify the no. of Iterations";
    iteration_option->_type = CA_ARG_INT;
    iteration_option->_value = &iterations;

    sampleArgs->AddOption(iteration_option);
    delete iteration_option;

    Option* buffer_option = new Option;
    CHECK_ALLOCATION(buffer_option, "Memory allocation error.\n");

    buffer_option->_sVersion = "";
    buffer_option->_lVersion = "dImage";
    buffer_option->_description =
        "Disable usage of OpenCL Image Object. Uses OpenCL Buffer instead";
    buffer_option->_type = CA_NO_ARGUMENT;
    buffer_option->_value = &dImageBuffer;

    sampleArgs->AddOption(buffer_option);
    delete buffer_option;

    return SDK_SUCCESS;
}

int
UnsharpMask::genBinaryImage()
{
    bifData binaryData;
    binaryData.kernelName = std::string("UnsharpMask_Kernels.cl");
    binaryData.flagsStr = std::string("");
    if(sampleArgs->isComplierFlagsSpecified())
    {
        binaryData.flagsFileName = std::string(sampleArgs->flags.c_str());
    }
    binaryData.binaryName = std::string(sampleArgs->dumpBinary.c_str());
    int status = generateBinaryImage(binaryData);
    return status;
}

int
UnsharpMask::setupUnsharpMask()
{
    sigma = radius/2.0f;
    imageOutputPtr = (unsigned char*)malloc(imageSize);
    origin[0] = origin[1] = origin[2] = 0;
    region[0] = width;
    region[1] = height;
    region[2] = 1;
    int dimen = 2*radius+1;

    try
    {
        float * gaussianPtr = (float*) queue.enqueueMapBuffer(gaussian1DBuffer, CL_TRUE,
                              CL_MAP_WRITE_INVALIDATE_REGION, 0, dimen*sizeof(float));
        generateGaussian1D(sigma, radius, gaussianPtr);
        queue.enqueueUnmapMemObject(gaussian1DBuffer, gaussianPtr);

        if(!dImageBuffer)
        {
            ptr = (unsigned char*) queue.enqueueMapImage(inputImageObj, CL_TRUE,
                    CL_MAP_WRITE_INVALIDATE_REGION, origin, region, &rowPitch, NULL);
            memcpy(ptr, inputImage.getPixels(), imageSize);
            queue.enqueueUnmapMemObject(inputImageObj, ptr);
        }
        else
        {
            float *gaussianPtr = (float*) queue.enqueueMapBuffer(gaussian2DBuffer, CL_TRUE,
                                 CL_MAP_WRITE_INVALIDATE_REGION, 0, dimen*dimen*sizeof(float));
            generateGaussian2D(sigma, radius, gaussianPtr);
            queue.enqueueUnmapMemObject(gaussian2DBuffer, gaussianPtr);
            ptr = (unsigned char*) queue.enqueueMapBuffer(inputBuffer, CL_TRUE,
                    CL_MAP_WRITE_INVALIDATE_REGION, 0, imageSize);
            memcpy(ptr, inputImage.getPixels(), imageSize);
            queue.enqueueUnmapMemObject(inputBuffer, ptr);
        }
    }
    catch (cl::Error e)
    {
        std::cout << e.what() << std::endl;
        return SDK_FAILURE;
    }

    return SDK_SUCCESS;
}

void
UnsharpMask::generateGaussian1D(float sigma, int radius, float* gaussianPtr )
{
    float sigmaSquare = sigma*sigma;
    int dimension = radius*2+1;
    float invSqr2PISigma = 1.0f/(sqrtf(2.0f*PI)*sigma);
    float twoSigmaSquare = 2.0f * sigma * sigma;
    float accumulatedGaussian = 0.0f;
    int x = -radius;

    for (int i = 0; i < (radius*2+1); i++,x++)
    {
        float g = invSqr2PISigma * expf(-((x*x)/twoSigmaSquare));
        gaussianPtr[i] = g;
        accumulatedGaussian += g;
    }
    for (int i = 0; i < (radius*2+1); i++,x++)
    {
        gaussianPtr[i] = gaussianPtr[i]/accumulatedGaussian;
    }
}

void
UnsharpMask::generateGaussian2D(float sigma, int radius, float* gaussianPtr)
{

    float sigmaSquare = sigma*sigma;
    int dimension = radius*2+1;

    float inv2PISigmaSquare = 1.0f / (2.0f*PI*sigmaSquare);
    float accumulatedGaussian = 0.0f;
    for (int y = 0; y < dimension; y++)
    {
        for (int x = 0; x < dimension; x++)
        {
            int tx = x - radius;
            int ty = y - radius;
            float g = inv2PISigmaSquare*::expf(-((tx*tx+ty*ty)/(2.0f*sigmaSquare)));
            gaussianPtr[y*dimension+x] = g;
            accumulatedGaussian+=g;
        }
    }
    for (int y = 0; y < dimension; y++)
    {
        for (int x = 0; x < dimension; x++)
        {
            gaussianPtr[y*dimension+x] = gaussianPtr[y*dimension+x]/accumulatedGaussian;
        }
    }
}

int
UnsharpMask::loadInputImage()
{
    inputImage.load(imageFile.c_str());

    if (!inputImage.isLoaded())
    {
        std::cout << "Failed to load image " << imageFile << std::endl;
        return SDK_FAILURE;
    }

    imageWidth= width = inputImage.width;
    imageHeight= height = inputImage.height;
    imageSize = height * width * sizeof(cl_uchar4);
    return SDK_SUCCESS;
}

int
UnsharpMask::setupCL()
{
    try
    {
        cl_int err = CL_SUCCESS;
        cl_device_type dType;

        if(sampleArgs->deviceType.compare("cpu") == 0)
        {
            dType = CL_DEVICE_TYPE_CPU;
        }
        else //deviceType == "gpu"
        {
            dType = CL_DEVICE_TYPE_GPU;
            if(sampleArgs->isThereGPU() == false)
            {
                std::cout << "GPU not found. Falling back to CPU device" << std::endl;
                dType = CL_DEVICE_TYPE_CPU;
            }
        }

        /*
        * Have a look at the available platforms and pick either
        * the AMD one if available or a reasonable default.
        */
        cl::Platform::get(&platforms);


        std::vector<cl::Platform>::iterator i;
        if(platforms.size() > 0)
        {
            if(sampleArgs->isPlatformEnabled())
            {
                i = platforms.begin() + sampleArgs->platformId;
            }
            else
            {
                for(i = platforms.begin(); i != platforms.end(); ++i)
                {
                    if(!strcmp((*i).getInfo<CL_PLATFORM_NAME>().c_str(),
                               "OpenCRun"))
                    {
                        break;
                    }
                }
            }
        }

        cl_context_properties cps[3] =
        {
            CL_CONTEXT_PLATFORM,
            (cl_context_properties)(*i)(),
            0
        };

        context = cl::Context(dType, cps, NULL, NULL);

        devices = context.getInfo<CL_CONTEXT_DEVICES>();

        std::cout << "Platform :" << (*i).getInfo<CL_PLATFORM_VENDOR>().c_str() << "\n";
        int deviceCount = (int)devices.size();
        int j = 0;
        for (std::vector<cl::Device>::iterator i = devices.begin(); i != devices.end();
                ++i, ++j)
        {
            std::cout << "Device " << j << " : ";
            std::string deviceName = (*i).getInfo<CL_DEVICE_NAME>();
            std::cout << deviceName.c_str() << "\n";
        }
        std::cout << "\n";

        if (deviceCount == 0)
        {
            std::cerr << "No device available\n";
            return SDK_FAILURE;
        }

        if(validateDeviceId(sampleArgs->deviceId, deviceCount))
        {
            error("validateDeviceId() failed");
            return SDK_FAILURE;
        }

        commandQueue = cl::CommandQueue(context, devices[sampleArgs->deviceId], 0);

        device.push_back(devices[sampleArgs->deviceId]);

        // create a CL program using the kernel source
        SDKFile kernelFile;
        std::string kernelPath = getPath();

        if(sampleArgs->isLoadBinaryEnabled())
        {
            kernelPath.append(sampleArgs->loadBinary.c_str());
            if(kernelFile.readBinaryFromFile(kernelPath.c_str()) != SDK_SUCCESS)
            {
                std::cout << "Failed to load kernel file : " << kernelPath << std::endl;
                return SDK_FAILURE;
            }
            cl::Program::Binaries programBinary(1,std::make_pair(
                                                    (const void*)kernelFile.source().data(),
                                                    kernelFile.source().size()));

            program = cl::Program(context, device, programBinary, NULL);
        }
        else
        {
            kernelPath.append("UnsharpMask_Kernels.cl");
            if(!kernelFile.open(kernelPath.c_str()))
            {
                std::cout << "Failed to load kernel file : " << kernelPath << std::endl;
                return SDK_FAILURE;
            }

            cl::Program::Sources programSource(1,
                                               std::make_pair(kernelFile.source().data(),
                                                       kernelFile.source().size()));

            program = cl::Program(context, programSource);
        }

        std::string flagsStr = std::string("");

        // Get additional options
        if(sampleArgs->isComplierFlagsSpecified())
        {
            SDKFile flagsFile;
            std::string flagsPath = getPath();
            flagsPath.append(sampleArgs->flags.c_str());
            if(!flagsFile.open(flagsPath.c_str()))
            {
                std::cout << "Failed to load flags file: " << flagsPath << std::endl;
                return SDK_FAILURE;
            }
            flagsFile.replaceNewlineWithSpaces();
            const char * flags = flagsFile.source().c_str();
            flagsStr.append(flags);
        }

        if(flagsStr.size() != 0)
        {
            std::cout << "Build Options are : " << flagsStr.c_str() << std::endl;
        }

        program.build(device, flagsStr.c_str());

        queue = cl::CommandQueue(context, devices[sampleArgs->deviceId], 0, &err);

        int dimen = 2*radius+1;
        cl::ImageFormat format(CL_BGRA,CL_UNSIGNED_INT8);

        if(loadInputImage()!=SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        gaussian1DBuffer = cl::Buffer(context,CL_MEM_READ_ONLY, dimen*sizeof(float));

        // create the 1D Gaussian kernel
        if(dImageBuffer)
        {
            gaussian2DBuffer = cl::Buffer(context,CL_MEM_READ_ONLY,
                                          dimen*dimen*sizeof(float));
            inputBuffer = cl::Buffer(context,CL_MEM_READ_ONLY, imageSize);
            outputBuffer = cl::Buffer (context,CL_MEM_WRITE_ONLY, imageSize);
            unsharp_mask_filter = cl::Kernel(program, "unsharp_mask_filter");

        }
        else
        {
            inputImageObj = cl::Image2D(context, CL_MEM_READ_ONLY, format, width, height);
            sharpenImageObj = cl::Image2D(context, CL_MEM_WRITE_ONLY, format, width,
                                          height);
            tmpImageObj = cl::Buffer(context,CL_MEM_READ_WRITE,
                                     width*height*sizeof(cl_float4));

            // Create kernel
            unsharp_mask_pass1 = cl::Kernel(program, "unsharp_mask_pass1");
            unsharp_mask_pass2 = cl::Kernel(program, "unsharp_mask_pass2");
        }
    }
    catch (cl::Error e)
    {
        if(e.err() == CL_BUILD_PROGRAM_FAILURE)
        {
            std::string str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[sampleArgs->deviceId]);

            std::cout << " \n\t\t\tBUILD LOG\n";
            std::cout << " ************************************************\n";
            std::cout << str << std::endl;
            std::cout << " ************************************************\n";
        }
        else
        {
            std::cout << e.what() << " failed!"<< std::endl;
            std::cout << "Error code: " << e.err() << std::endl;
        }
        return SDK_FAILURE;
    }
    return SDK_SUCCESS;
}

int
UnsharpMask::setup()
{
    if(iterations < 1)
    {
        std::cout<<"Error, iterations cannot be 0 or negative. Exiting..\n";
        return SDK_FAILURE;
    }
    // create and initialize timers
    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    if(setupCL() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(setupUnsharpMask() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    sampleTimer->stopTimer(timer);
    // Compute setup time
    setupTime = (double)(sampleTimer->readTimer(timer));

    return SDK_SUCCESS;

}

int
UnsharpMask::runCLKernels()
{
    int status;

    try
    {
        if(!dImageBuffer)
        {
            // Set kernel arguments for 1st pass
            status = unsharp_mask_pass1.setArg(0, inputImageObj);
            status = unsharp_mask_pass1.setArg(1, tmpImageObj);
            status = unsharp_mask_pass1.setArg(2, width);
            status = unsharp_mask_pass1.setArg(3, gaussian1DBuffer);
            status = unsharp_mask_pass1.setArg(4, radius);

            //enqueue the two kernels
            cl::NDRange globalThreads(width,height);
            status = queue.enqueueNDRangeKernel(unsharp_mask_pass1,cl::NullRange,
                                                globalThreads);

            // Set kernel arguments 2nd pass
            status = unsharp_mask_pass2.setArg(0, inputImageObj);
            status = unsharp_mask_pass2.setArg(1, tmpImageObj);
            status = unsharp_mask_pass2.setArg(2, sharpenImageObj);
            status = unsharp_mask_pass2.setArg(3, width);
            status = unsharp_mask_pass2.setArg(4, height);
            status = unsharp_mask_pass2.setArg(5, gaussian1DBuffer);
            status = unsharp_mask_pass2.setArg(6, radius);
            status = unsharp_mask_pass2.setArg(7, threshold);
            status = unsharp_mask_pass2.setArg(8, amount);

            status = queue.enqueueNDRangeKernel(unsharp_mask_pass2,cl::NullRange,
                                                globalThreads);
        }
        else
        {
            // Set kernel arguments 2nd pass
            status = unsharp_mask_filter.setArg(0, inputBuffer);
            status = unsharp_mask_filter.setArg(1, outputBuffer);
            status = unsharp_mask_filter.setArg(2, width);
            status = unsharp_mask_filter.setArg(3, height);
            status = unsharp_mask_filter.setArg(4, gaussian2DBuffer);
            status = unsharp_mask_filter.setArg(5, radius);
            status = unsharp_mask_filter.setArg(6, threshold);
            status = unsharp_mask_filter.setArg(7, amount);
            cl::NDRange globalThreads(width,height);

            status = queue.enqueueNDRangeKernel(unsharp_mask_filter,cl::NullRange,
                                                globalThreads);
        }
    }
    catch (cl::Error e)
    {
        std::cout << e.what() << std::endl;
        std::cout << "Error code: " << e.err() << std::endl;
        return SDK_FAILURE;
    }
    return SDK_SUCCESS;
}

int
UnsharpMask::run()
{
    std::cout << "Executing kernel for " << iterations <<
              " iterations" <<std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    // create and initialize timers
    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    try
    {
        for(int i = 0; i < iterations; i++)
        {
            // Set kernel arguments and run kernel
            if(runCLKernels() != SDK_SUCCESS)
            {
                return SDK_FAILURE;
            }
        }
    }
    catch (cl::Error e)
    {
        std::cout << e.what() << std::endl;
        std::cout << "Error code: " << e.err() << std::endl;
        return SDK_FAILURE;
    }

    queue.finish();

    sampleTimer->stopTimer(timer);

    // Compute kernel time
    kernelTime = (double)(sampleTimer->readTimer(timer))/ iterations;
    return SDK_SUCCESS;

}

void
UnsharpMask::unsharpMask2PassCPU(unsigned char* input, unsigned char* output
                                 , int width, int height
                                 , float* gaussianPtr, int gaussianPtrRadius
                                 , float threshold, float amount)
{

    float* tmp = (float*)malloc(width*height*3*sizeof(float));
    float* pTmp = tmp;

    // 1D Guassian, horizontal
    for (int j = 0; j < height; j++)
    {
        for (int i = 0; i < width; i++)
        {
            float gb,gg,gr;
            gb = gg = gr = 0.0f;
            int tx = i - gaussianPtrRadius;
            for (int n = 0; n <= gaussianPtrRadius*2; n++,tx++)
            {
                Pixel currentPixel = currentPixel = getPixel(input, CLAMP(tx, 0, width-1), j,
                                                    width);
                float weight = gaussianPtr[n];
                gb+= weight*(float)currentPixel.b;
                gg+= weight*(float)currentPixel.g;
                gr+= weight*(float)currentPixel.r;
            }
            *pTmp++ = gb;
            *pTmp++ = gg;
            *pTmp++ = gr;
        }
    }
    // 1D Guassian, vertical
    for (int j = 0; j < height; j++)
    {
        for (int i = 0; i < width; i++)
        {
            float gb,gg,gr;
            gb = gg = gr = 0.0f;

            int ty = j - gaussianPtrRadius;
            for (int n = 0; n <= gaussianPtrRadius*2; n++,ty++)
            {

                int clampedY = CLAMP(ty, 0, height-1);
                pTmp = tmp + (clampedY * width + i) * 3;
                float weight = gaussianPtr[n];

                gb+= weight **pTmp++;
                gg+= weight **pTmp++;
                gr+= weight **pTmp;
            }

            // diff pixel
            Pixel currentPixel = getPixel(input, i, j, width);
            float cp_b = currentPixel.b;
            float cp_g = currentPixel.g;
            float cp_r = currentPixel.r;

            float diff_b = cp_b-gb;
            cp_b += (::fabsf(diff_b) > threshold) ? diff_b*amount : 0.0f;

            float diff_g = cp_g-gg;
            cp_g += (::fabsf(diff_g) > threshold) ? diff_g*amount : 0.0f;

            float diff_r = cp_r-gr;
            cp_r += (::fabsf(diff_r) > threshold) ? diff_r*amount : 0.0f;

            currentPixel.b = (unsigned char)CLAMP(cp_b+0.5f, 0.0f, 255.0f);
            currentPixel.g = (unsigned char)CLAMP(cp_g+0.5f, 0.0f, 255.0f);
            currentPixel.r = (unsigned char)CLAMP(cp_r+0.5f, 0.0f, 255.0f);

            setPixel(output, i, j, width, currentPixel);
        }
    }
}

int
UnsharpMask::verifyResults()
{
    try
    {
        if(!dImageBuffer)
        {
            unsharpMaskImage = (unsigned char*) queue.enqueueMapImage(sharpenImageObj,
                               CL_TRUE, CL_MAP_READ, origin, region, &rowPitch, NULL);
        }
        else
        {
            unsharpMaskImage = (unsigned char*) queue.enqueueMapBuffer(outputBuffer,
                               CL_TRUE, CL_MAP_READ, 0, imageSize);
        }

    }
    catch (cl::Error e)
    {
        std::cout << e.what() << std::endl;
        std::cout << "Error code: " << e.err() << std::endl;
        return SDK_FAILURE;
    }

    if(sampleArgs->verify)
    {
        int radius = UnsharpMask::radius;
        int dimen = 2*radius+1;
        float sigma = UnsharpMask::sigma;
        cpuUnsharpMaskImage = (unsigned char*) malloc(imageSize);
        unsigned char* input = (unsigned char*)inputImage.getPixels();
        float* gaussianPtr;
        gaussianPtr = (float*) queue.enqueueMapBuffer(gaussian1DBuffer, CL_TRUE,
                      CL_MAP_READ, 0, dimen*sizeof(float));

        for (int i = 0; i < iterations; i++)
        {
            unsharpMask2PassCPU(input, cpuUnsharpMaskImage
                                , width, height
                                , gaussianPtr, radius
                                , threshold,amount);
        }

        queue.enqueueUnmapMemObject(gaussian1DBuffer, gaussianPtr);
        for (int j = 0; j < height; j++)
        {
            for (int i = 0; i < width; i++)
            {
                Pixel oclPixel = getPixel(unsharpMaskImage, i, j, width);
                Pixel cpuPixel = getPixel(cpuUnsharpMaskImage, i, j, width);
                if (oclPixel != cpuPixel)
                {
                    std::cout << "Error at (" << i << "," << j << ")\t";
                    std::cout << "CPU: (" << (unsigned int) cpuPixel.b << "," <<
                              (unsigned int) cpuPixel.g << "," << (unsigned int) cpuPixel.r << ")\t";
                    std::cout << "OpenCL: (" << (unsigned int) oclPixel.b << "," <<
                              (unsigned int) oclPixel.g << "," << (unsigned int) oclPixel.r << ")" <<
                              std::endl;
                    std::cout << "Failed\n" << std::endl;
                    return SDK_FAILURE;
                }
            }
        }

        std::cout << "Passed!\n" << std::endl;

        // write cpu output to the image file
        memcpy(inputImage.getPixels(), cpuUnsharpMaskImage, imageSize);
        inputImage.write(CPU_OUTPUT_IMAGE);
    }

    // write GPU output image to a file
    memcpy(inputImage.getPixels(), unsharpMaskImage, imageSize);
    inputImage.write(OUTPUT_IMAGE);

    try
    {
        memcpy(imageOutputPtr, unsharpMaskImage, imageSize);
        if(!dImageBuffer)
        {
            queue.enqueueUnmapMemObject(sharpenImageObj, unsharpMaskImage);
        }
        else
        {
            queue.enqueueUnmapMemObject(outputBuffer, unsharpMaskImage);
        }
    }
    catch (cl::Error e)
    {
        std::cout << e.what() << std::endl;
        std::cout << "Error code: " << e.err() << std::endl;
        return SDK_FAILURE;
    }

    return SDK_SUCCESS;
}

int
UnsharpMask::cleanup()
{
    // release program resources (input memory etc.)
    FREE(cpuUnsharpMaskImage);
    return SDK_SUCCESS;
}

void
UnsharpMask::printStats()
{
    if(sampleArgs->timing)
    {
        std::string strArray[5] =
        {
            "Width",
            "Height",
            "Setup Time(sec)",
            "Avg. Kernel Time(sec)",
            "Pixels/sec"

        };
        std::string stats[5];

        stats[0] = toString(width, std::dec);
        stats[1] = toString(height, std::dec);
        stats[2] = toString(setupTime, std::dec);
        stats[3] = toString(kernelTime, std::dec);
        stats[4] = toString((width * height)/kernelTime ,std::dec);

        printStatistics(strArray, stats, 5);
    }
}

void
GLInit()
{
    glClearColor(0.0 ,0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT);
    glClear(GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
}

void
displayFunc()
{
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawPixels(imageWidth,imageHeight, GL_RGBA, GL_UNSIGNED_BYTE, imageOutputPtr);
    glFlush();
    glutSwapBuffers();
}

void
usage()
{
    std::cout << "Press '+' or 'a' to increase the number of points." << std::endl;
    std::cout << "Press '-' or 'z' to decrease the number of points." << std::endl;
    std::cout << "Press 'q' to quit." << std::endl;
    std::cout << "Press 'h' to show this message." << std::endl;
}

void
keyboardFunc (unsigned char key, int mouseX, int mouseY)
{
    switch ( key )
    {
    case 'q':
        exit (0);
        break;
    case '+':
        glutPostRedisplay();
        break;
    case '-':
        glutPostRedisplay();
        break;
    case 'h':
        usage();
        break;
    default:
        break;
    }
}

void UnsharpMask::runGUI(int argc , char *argv[])
{
    // Run in  graphical window if requested
    glutInit(&argc, argv);
    glutInitWindowPosition(100,10);
    glutInitWindowSize(1024,1024);
    glutInitDisplayMode( GLUT_RGB | GLUT_DOUBLE );
    glutCreateWindow("Unsharp Mask Filter");
    GLInit();
    glutDisplayFunc(displayFunc);
    ::usage();
    glutKeyboardFunc(keyboardFunc);
    glutMainLoop();
}

int
main(int argc, char *argv[])
{

    UnsharpMask clUnsharpMask;

    if(clUnsharpMask.initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(clUnsharpMask.sampleArgs->parseCommandLine(argc, argv))
    {
        return SDK_FAILURE;
    }

    if(clUnsharpMask.sampleArgs->isDumpBinaryEnabled())
    {
        return clUnsharpMask.genBinaryImage();
    }

    if(clUnsharpMask.setup() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(clUnsharpMask.run() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(clUnsharpMask.verifyResults() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(clUnsharpMask.cleanup() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    clUnsharpMask.printStats();

    if(clUnsharpMask.displayImage())
    {
        clUnsharpMask.runGUI(argc,argv);
    }

    return SDK_SUCCESS;
}
