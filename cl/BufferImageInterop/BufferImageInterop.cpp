/**********************************************************************
Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/
#include "BufferImageInterop.hpp"

int BufferImageInterop::initialize()
{
    // Call base class Initialize to get default configuration
    if(sampleArgs->initialize())
    {
        return SDK_FAILURE;
    }

    Option* num_iterations = new Option;
    CHECK_ALLOCATION(num_iterations, "Memory allocation error.\n");

    num_iterations->_sVersion = "i";
    num_iterations->_lVersion = "iterations";
    num_iterations->_description = "Number of iterations for kernel execution";
    num_iterations->_type = CA_ARG_INT;
    num_iterations->_value = &iterations;

    sampleArgs->AddOption(num_iterations);

    if( iterations <= 0)
    {
        std::cout << "\nError: Number of iterations should be greater than 0!";
        exit(SDK_EXPECTED_FAILURE);
    }

    delete num_iterations;

    return SDK_SUCCESS;

}

int
BufferImageInterop::genBinaryImage()
{
    bifData binaryData;
    binaryData.kernelName = std::string("BufferImageInterop_Kernels.cl");
    binaryData.flagsStr = std::string("");
    if(sampleArgs->isComplierFlagsSpecified())
    {
        binaryData.flagsFileName = std::string(sampleArgs->flags.c_str());
    }
    binaryData.binaryName = std::string(sampleArgs->dumpBinary.c_str());
    int status = generateBinaryImage(binaryData);
    return status;
}

int BufferImageInterop::readInputImage(std::string inputImageName)
{
    // load input bitmap image
    std::string filePath = getPath() + inputImageName;
    inputBitmap.load(filePath.c_str());
    if(!inputBitmap.isLoaded())
    {
        std::cout << "Failed to load input image!";
        return SDK_FAILURE;
    }

    // get width and height of input image
    height = inputBitmap.getHeight();
    width = inputBitmap.getWidth();

    // allocate memory for output image data
    outputImageData = (cl_uchar4*)malloc(width * height * sizeof(cl_uchar4));
    CHECK_ALLOCATION(outputImageData,
                     "Failed to allocate memory! (outputImageData)");

    // initializa the Image data to NULL
    memset(outputImageData, 0, width * height * pixelSize);

    // get the pointer to pixel data
    pixelData = inputBitmap.getPixels();

    if(pixelData == NULL)
    {
        std::cout << "Failed to read pixel Data!";
        return SDK_FAILURE;
    }

    // allocate memory for verification output
    verificationInput = (uchar4*)malloc(width * height * pixelSize);
    CHECK_ALLOCATION(verificationInput,
                     "verificationOutput heap allocation failed!");

    // initialize the data to NULL
    memcpy(verificationInput, pixelData, width * height * pixelSize);

    // allocate memory for verification output
    verificationOutput = (uchar4*)malloc(width * height * pixelSize);
    CHECK_ALLOCATION(verificationOutput,
                     "verificationOutput heap allocation failed!");

    // initialize the data to NULL
    memset(verificationOutput, 0, width * height * pixelSize);

    return SDK_SUCCESS;
}

int BufferImageInterop::writeOutputImage(std::string outputImageName)
{
    // copy output image data back to original pixel data
    memcpy(pixelData, outputImageData, width * height * pixelSize);

    // write the output bmp file
    if(!inputBitmap.write(outputImageName.c_str()))
    {
        std::cout << "Failed to write output image!";
        return SDK_FAILURE;
    }
    return SDK_SUCCESS;
}


template<typename T>
int BufferImageInterop::mapBuffer(cl_mem deviceBuffer, T* &hostPointer,
                                  size_t sizeInBytes, cl_map_flags flags)
{
    cl_int status;
    hostPointer = (T*) clEnqueueMapBuffer(commandQueue,
                                          deviceBuffer,
                                          CL_TRUE,
                                          flags,
                                          0,
                                          sizeInBytes,
                                          0,
                                          NULL,
                                          NULL,
                                          &status);
    CHECK_OPENCL_ERROR(status, "clEnqueueMapBuffer failed");

    return SDK_SUCCESS;
}

int
BufferImageInterop::unmapBuffer(cl_mem deviceBuffer, void* hostPointer)
{
    cl_int status;
    status = clEnqueueUnmapMemObject(commandQueue,
                                     deviceBuffer,
                                     hostPointer,
                                     0,
                                     NULL,
                                     NULL);
    CHECK_OPENCL_ERROR(status, "clEnqueueUnmapMemObject failed");

    return SDK_SUCCESS;
}

int BufferImageInterop::setupCL()
{
    cl_int status = 0;
    cl_device_type dType;

    if(sampleArgs->deviceType.compare("cpu") == 0)
    {
        dType = CL_DEVICE_TYPE_CPU;
    }
    else //deviceType = "gpu"
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

    cl_platform_id platform = NULL;
    int retValue = getPlatform(platform, sampleArgs->platformId,
                               sampleArgs->isPlatformEnabled());
    CHECK_ERROR(retValue, SDK_SUCCESS, "getPlatform() failed");

    // Display available devices.
    retValue = displayDevices(platform, dType);
    CHECK_ERROR(retValue, SDK_SUCCESS, "displayDevices() failed");


    // If we could find our platform, use it. Otherwise use just available platform.

    cl_context_properties cps[3] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platform,
        0
    };

    context = clCreateContextFromType(
                  cps,
                  dType,
                  NULL,
                  NULL,
                  &status);
    CHECK_OPENCL_ERROR( status, "clCreateContextFromType failed.");

    // getting device on which to run the sample
    status = getDevices(context, &devices, sampleArgs->deviceId,
                        sampleArgs->isDeviceIdEnabled());
    CHECK_ERROR(status, SDK_SUCCESS, "getDevices() failed");


    retValue = deviceInfo.setDeviceInfo(devices[sampleArgs->deviceId]);
    CHECK_ERROR(retValue, 0, "SDKDeviceInfo::setDeviceInfo() failed");

    // if((strstr(deviceInfo.extensions,
    //            "cl_khr_image2d_from_buffer_read_only") == NULL) &&
    //         (strstr(deviceInfo.extensions, "cl_khr_image2d_from_buffer") == NULL)
    //         && (strstr(deviceInfo.extensions,
    //                    "cl_amd_image2d_from_buffer_read_only") == NULL)
    //  
    //   )
    // {
    //     std::cout <<
    //               "\nError: Selected device doesn't support Buffer-Image \
    //                          Interop(cl_amd_image2d_from_buffer)";
    //     exit(SDK_EXPECTED_FAILURE);
    // }


    // The block is to move the declaration of prop closer to its use
    cl_command_queue_properties prop = 0;
    commandQueue = clCreateCommandQueue(
                       context,
                       devices[sampleArgs->deviceId],
                       prop,
                       &status);
    CHECK_OPENCL_ERROR( status, "clCreateCommandQueue failed.");


// Create buffer object for input image and output image.
    inputImageBuffer = clCreateBuffer(context,
                                      CL_MEM_READ_ONLY ,
                                      width * height * pixelSize, NULL, &status);
    CHECK_OPENCL_ERROR(status, "clCreateBuffer failed. (inputImageBuffer)");

    outputImageBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                       width * height * pixelSize, 0, &status);
    CHECK_OPENCL_ERROR(status, "clCreateBuffer failed. (outputImageBuffer)");

    status = mapBuffer<cl_uchar4>(inputImageBuffer,inputImageData,
                                  width * height * sizeof(cl_uchar4),CL_MAP_WRITE);

    memcpy(inputImageData, pixelData, width * height *  pixelSize);

    status = unmapBuffer(inputImageBuffer,inputImageData);

// Create input and output image objects.
    imageFormat.image_channel_order = CL_RGBA;
    imageFormat.image_channel_data_type = CL_UNSIGNED_INT8;

    imageDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    imageDesc.image_width = width;
    imageDesc.image_height = height;

    /**********************Image - Buffer Interop**************************/
    // imageDesc.buffer = outputImageBuffer;
    imageDesc.buffer = NULL;
    imageDesc.image_depth = 1;
    imageDesc.image_array_size = 0;
    imageDesc.image_row_pitch = 0;
    imageDesc.image_slice_pitch = 0;
    imageDesc.num_mip_levels = 0;
    imageDesc.num_samples = 0;

    inputImage = clCreateImage(context,
                               CL_MEM_READ_ONLY,
                               &imageFormat,
                               &imageDesc,/*Buffer passed as argument */
                               NULL,
                               &status);

    CHECK_OPENCL_ERROR(status, "clCreateImage failed. (inputImage)");

    imageDesc.buffer= NULL;

    outputImage = clCreateImage(context, CL_MEM_WRITE_ONLY, &imageFormat,
                                &imageDesc, 0, &status);
    CHECK_OPENCL_ERROR(status, "clCreateImage failed. (outputImage)");

    // Build the kernels.
    buildProgramData buildData;
    buildData.kernelName = std::string("BufferImageInterop_kernels.cl");
    buildData.devices = devices;
    buildData.deviceId = 0;
    status = buildOpenCLProgram(program, context, buildData);
    CHECK_ERROR(status, 0, "sampleCommand::buildOpenCLProgram() failed.");

    sepiaKernel = clCreateKernel(program, "sepiaToning", &status);
    CHECK_OPENCL_ERROR(status, "clCreateKernel failed (sepiaToning).");

    imageReverseKernel = clCreateKernel(program, "imageReverse", &status);
    CHECK_OPENCL_ERROR(status, "clCreateKernel failed (imageReverse).");

    return SDK_SUCCESS;
}

int BufferImageInterop::copyBufferToImage(cl_mem Buffer, cl_mem Image)
{
    int status;
    const size_t dst_origin[3] = { 0, 0, 0};
    const size_t dst_region[3] = { width, height, 1};

    clEnqueueCopyBufferToImage(commandQueue,
                               Buffer,
                               Image,
                               0,
                               dst_origin,
                               dst_region,
                               0,
                               NULL,
                               NULL);
    CHECK_OPENCL_ERROR(status, "clEnqueueCopyBufferToImage failed");

    return SDK_SUCCESS;
}

int BufferImageInterop::setup()
{
    // Allocate host memory and read input image
    if(readInputImage(INPUT_IMAGE) != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // Set up OpenCL  and sample related environment.
    if (setupCL() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    return SDK_SUCCESS;
}


int BufferImageInterop::runCLKernels()
{
    cl_int status;
    cl_event ndrEvent;
    size_t globalThreads[] = {width, height};
    size_t localThreads[] = {16, 16};

    // Sepia Kernel
    // Set the arguments and run the sepaiaKernel.
    status = clSetKernelArg(sepiaKernel, 0, sizeof(cl_mem), &inputImageBuffer);
    CHECK_OPENCL_ERROR(status, "clSetKernelArg failed (inputImageBuffer).");

    status = clSetKernelArg(sepiaKernel, 1, sizeof(cl_mem), &outputImageBuffer);
    CHECK_OPENCL_ERROR(status, "clSetKernelArg failed (outputImageBuffer).");

    // Sepia Kernel
    status = clEnqueueNDRangeKernel(commandQueue, sepiaKernel, 2, NULL,
                                    globalThreads, NULL, 0, NULL, &ndrEvent);

    CHECK_OPENCL_ERROR(status, "clEnqueueNDRangeKernel failed (sepiaToning).");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed - sepia.");

    status = waitForEventAndRelease(&ndrEvent);
    CHECK_OPENCL_ERROR(status, "waitForEventAndRelease failed - sepia (ndrEvent).");

    // Move intermediate output data to the input image object for the imageReverseKernel
    status = copyBufferToImage(outputImageBuffer, inputImage);
    CHECK_OPENCL_ERROR(status, "copyBufferToImage failed - sepia (ndrEvent).");

    // Set the arguments and run the imageReverseKernel.
    status = clSetKernelArg(imageReverseKernel, 0, sizeof(cl_mem), &inputImage);
    CHECK_OPENCL_ERROR(status, "clSetKernelArg failed (inputImage).");

    status = clSetKernelArg(imageReverseKernel, 1, sizeof(cl_mem), &outputImage);
    CHECK_OPENCL_ERROR(status, "clSetKernelArg failed (outputImage).");

    //Image reverse kernel
    status = clEnqueueNDRangeKernel(commandQueue, imageReverseKernel, 2, NULL,
                                    globalThreads, NULL, 0, NULL, &ndrEvent);
    CHECK_OPENCL_ERROR(status,
                       "clEnqueueNDRangeKernel failed (imageReverseKernel).");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed - imageReverse.");

    status = waitForEventAndRelease(&ndrEvent);
    CHECK_OPENCL_ERROR(status,
                       "waitForEventAndRelease failed - imageReverse (ndrEvent).");

    return SDK_SUCCESS;
}


int BufferImageInterop::run()
{
    // Warm up
    for(int i = 0; i < 2 && iterations != 1; i++)
    {
        // Arguments are set and execution call is enqueued on command buffer
        if(runCLKernels() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }
    }

    std::cout << "Executing kernel for " << iterations <<
              " iterations" << std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    for(int i = 0; i < iterations; i++)
    {
        // Arguments are set and execution call is enqueued on command buffer
        if(runCLKernels() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }
    }

    sampleTimer->stopTimer(timer);
    totalKernelTime = (double)(sampleTimer->readTimer(timer));

    return SDK_SUCCESS;
}

void BufferImageInterop::printStats()
{
    if(sampleArgs->timing)
    {

        std::string strArray[4] = {"Width", "Height", "Avg. Kernel Time (sec)", "Pixels/sec"};
        std::string stats[4];

        sampleTimer->totalTime = ( totalKernelTime / iterations );

        stats[0] = toString(width   , std::dec);
        stats[1] = toString(height, std::dec);
        stats[2] = toString( sampleTimer->totalTime, std::dec);
        stats[3] = toString((width * height)/sampleTimer->totalTime, std::dec);

        printStatistics(strArray, stats, 4);

    }
}

int BufferImageInterop::cleanup()
{
    /* Releases OpenCL resources (Context, Memory etc.) */
    cl_int status;

    status = clReleaseKernel(sepiaKernel);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.");

    status = clReleaseProgram(program);
    CHECK_OPENCL_ERROR(status, "clReleaseProgram failed.");

    status = clReleaseMemObject(inputImageBuffer);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.");

    status = clReleaseMemObject(outputImageBuffer);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.");

    status = clReleaseMemObject(inputImage);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.");

    status = clReleaseMemObject(outputImage);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.");

    status = clReleaseCommandQueue(commandQueue);
    CHECK_OPENCL_ERROR(status, "clReleaseCommandQueue failed.");

    status = clReleaseContext(context);
    CHECK_OPENCL_ERROR(status, "clReleaseContext failed.");

    // release program resources (input memory etc.)

    FREE(devices);

    FREE(verificationInput);

    FREE(verificationOutput);

    return SDK_SUCCESS;
}


int BufferImageInterop::CPUReference()
{

    for (unsigned int index = 0; index < width * height ; index++)
    {
        int red, green, blue;

        red = verificationInput[index].x;
        green = verificationInput[index].y;
        blue = verificationInput[index].z;


        double grayscale =(0.3 * red + 0.59 * green + 0.11 * blue);

        double depth = 1.8;

        red = (int)(grayscale + depth * 56.6 );
        if (red > 255)
        {
            red = 255;
        }

        green = (int)(grayscale + depth * 33.3 );
        if (green > 255)
        {
            green = 255;
        }

        blue = (int)(grayscale + depth * 10.1);

        if (blue > 255)
        {
            blue = 255;
        }

        //Index caluclation for reversing the image
        int x= index % width;
        int y =index / height;

        int temp = (width -1 -x ) + width *y;

        verificationOutput[temp].x = red;
        verificationOutput[temp].y = green;
        verificationOutput[temp].z = blue;
        verificationOutput[temp].w = verificationInput[index].w;

    }

    return SDK_SUCCESS;
}


int BufferImageInterop::verifyResults()
{

    cl_int status;
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {width, height, 1};
    cl_event rdEvent;

    status = clEnqueueReadImage(commandQueue, outputImage, CL_TRUE, origin, region,
                                0, 0, outputImageData, 0, NULL,&rdEvent);

    CHECK_OPENCL_ERROR(status, "clEnqueueReadImage failed - imageReverse");

    status = waitForEventAndRelease(&rdEvent);

    CHECK_OPENCL_ERROR(status, "waitForEventAndRelease failed - rdEvent");

    // write image
    if (writeOutputImage("BufferImageInterop_Output.bmp") != SDK_SUCCESS)
    {
        std::cout << "writing image output failed." << std::endl;
        return SDK_FAILURE;
    }

    if(!sampleArgs->verify)
    {
        return SDK_SUCCESS;
    }

    if(CPUReference() == SDK_FAILURE)
    {
        return SDK_FAILURE;
    }

    for (unsigned int x= 0 ; x < width; x++)
    {
        for (unsigned int y =0 ; y < height ; y++)
        {
            int idx = y + height * x;
            if(     abs(pixelData[idx].x - verificationOutput[idx].x) > 1 ||
                    abs(pixelData[idx].y - verificationOutput[idx].y) > 1 ||
                    abs(pixelData[idx].w - verificationOutput[idx].w) > 1 ||
                    abs(pixelData[idx].z - verificationOutput[idx].z) > 1
              )
            {
                std::cout << "Error: Data mismatch at pixel (" << y <<"," << x << ")\n";
                std::cout << "\nFailed!\n";
                std::cout << (unsigned int )pixelData[idx].x << " " << (unsigned int )
                          verificationOutput[idx].x << "\n";
                std::cout << (unsigned int )pixelData[idx].y << " " << (unsigned int )
                          verificationOutput[idx].y << "\n";
                std::cout << (unsigned int )pixelData[idx].z << " " << (unsigned int )
                          verificationOutput[idx].z << "\n";
                std::cout << (unsigned int )pixelData[idx].w << " " << (unsigned int )
                          verificationOutput[idx].w << "\n";
                return SDK_FAILURE;
            }

        }
    }
    std::cout << "\nPassed!\n";
    return SDK_SUCCESS;
}


int
main(int argc, char * argv[])
{
    BufferImageInterop clBufferImageInterop;

    if(clBufferImageInterop.initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(clBufferImageInterop.sampleArgs->parseCommandLine(argc, argv) != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(clBufferImageInterop.sampleArgs->isDumpBinaryEnabled())
    {
        return clBufferImageInterop.genBinaryImage();
    }
    else
    {
        if(clBufferImageInterop.setup() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        if(clBufferImageInterop.run() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        if(clBufferImageInterop.verifyResults() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        if(clBufferImageInterop.cleanup() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

        clBufferImageInterop.printStats();
    }

    return SDK_SUCCESS;
}
