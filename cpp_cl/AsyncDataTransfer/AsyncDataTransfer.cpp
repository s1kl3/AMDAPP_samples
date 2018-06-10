/**********************************************************************
Copyright ©2012 Advanced Micro Devices, Inc. All rights reserved.

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

#include "AsyncDataTransfer.hpp"

int AsyncDataTransfer::initialize()
{
    // Call base class Initialize to get default configuration
    CHECK_ERROR(sampleArgs->initialize(), SDK_SUCCESS,
                "OpenCL resource initilization failed");
    Option* buffer_size = new Option;
    CHECK_ALLOCATION(buffer_size, "Memory allocation error.\n");

    buffer_size->_sVersion = "x";
    buffer_size->_lVersion = "size";
    buffer_size->_description = "Buffer size in Bytes per Kernel";
    buffer_size->_type = CA_ARG_INT;
    buffer_size->_value = &bufferSize;

    sampleArgs->AddOption(buffer_size);
    delete buffer_size;

    Option* num_iterations = new Option;
    CHECK_ALLOCATION(num_iterations, "Memory allocation error.\n");

    num_iterations->_sVersion = "i";
    num_iterations->_lVersion = "iterations";
    num_iterations->_description = "Number of iterations for kernel execution";
    num_iterations->_type = CA_ARG_INT;
    num_iterations->_value = &iterations;

    sampleArgs->AddOption(num_iterations);
    delete num_iterations;

    return SDK_SUCCESS;
}

int AsyncDataTransfer::genBinaryImage()
{
    bifData binaryData;
    binaryData.kernelName = std::string("AsyncDataTransfer_Kernels.cl");
    binaryData.flagsStr = std::string("");
    if(sampleArgs->isComplierFlagsSpecified())
    {
        binaryData.flagsFileName = std::string(sampleArgs->flags.c_str());
    }
    binaryData.binaryName = std::string(sampleArgs->dumpBinary.c_str());
    int status = generateBinaryImage(binaryData);
    return status;
}

int AsyncDataTransfer::setupCL(void)
{
    int status = 0;
    cl_device_type dType;

    if(sampleArgs-> deviceType.compare("cpu") == 0)
    {
        dType = CL_DEVICE_TYPE_CPU;
    }
    else //sampleArgs-> deviceType = "gpu"
    {
        dType = CL_DEVICE_TYPE_GPU;
        if(sampleArgs-> isThereGPU() == false)
        {
            std::cout << "GPU not found. Falling back to CPU device" << std::endl;
            dType = CL_DEVICE_TYPE_CPU;
        }
    }

    // Get platform
    cl_platform_id platform = NULL;
    status = getPlatform(platform, sampleArgs->platformId,
                         sampleArgs->isPlatformEnabled());
    CHECK_ERROR(status, SDK_SUCCESS, "getPlatform() failed");

    // Display available devices.
    status = displayDevices(platform, dType);
    CHECK_ERROR(status, SDK_SUCCESS, "displayDevices() failed");

    // If we could find our platform, use it. Otherwise use just available platform.
    cl_context_properties cps[3] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platform,
        0
    };

    context = cl::Context(dType, cps, NULL, NULL, &status);
    CHECK_OPENCL_ERROR(status, "Context::Context() failed.");

    devices = context.getInfo<CL_CONTEXT_DEVICES>();
    CHECK_ERROR((devices.size() > 0), true, "No device available");

    if(validateDeviceId(sampleArgs-> deviceId, (int)devices.size()))
    {
        error("validateDeviceId() failed");
        return SDK_FAILURE;
    }

    // Set device info of given sampleArgs-> deviceId
    status = setDeviceInfo(devices[sampleArgs-> deviceId]);
    CHECK_ERROR(status, SDK_SUCCESS, "SDKDeviceInfo::setDeviceInfo() failed");

    // Create queues(Kernel, Read & Write)
    kernelQueue = cl::CommandQueue(context, devices[sampleArgs-> deviceId], 0,
                                   &status);
    CHECK_OPENCL_ERROR(status, "cl::CommandQueue(queueKernel) failed.");

    readQueue = cl::CommandQueue(context, devices[sampleArgs-> deviceId], 0,
                                 &status);
    CHECK_OPENCL_ERROR(status, "cl::CommandQueue(queueRead) failed.");

    writeQueue = cl::CommandQueue(context, devices[sampleArgs-> deviceId], 0,
                                  &status);
    CHECK_OPENCL_ERROR(status, "cl::CommandQueue(queueWrite) failed.");

    // Create device buffers
    inputBuffer     = new cl::Buffer[numKernels];
    outputBuffer    = new cl::Buffer[numKernels];
    inPrepinBuffer  = new cl::Buffer[numKernels];
    outPrepinBuffer = new cl::Buffer[numKernels];
    for(int k = 0; k < numKernels; ++k)
    {
        inputBuffer[k] = cl::Buffer(context, CL_MEM_READ_ONLY, bufferSize, NULL,
                                    &status);
        CHECK_OPENCL_ERROR(status, "cl::Buffer(inputBuffer) failed.");

        outputBuffer[k] = cl::Buffer(context, CL_MEM_WRITE_ONLY, bufferSize, NULL,
                                     &status);
        CHECK_OPENCL_ERROR(status, "cl::Buffer(outputBuffer) failed.");

        inPrepinBuffer[k] = cl::Buffer(context, CL_MEM_ALLOC_HOST_PTR, bufferSize, NULL,
                                       &status);
        CHECK_OPENCL_ERROR(status, "cl::Buffer(inPrepinBuffer) failed.");

        outPrepinBuffer[k] = cl::Buffer(context, CL_MEM_ALLOC_HOST_PTR, bufferSize,
                                        NULL, &status);
        CHECK_OPENCL_ERROR(status, "cl::Buffer(outprepinBuffer) failed.");
    }

    // initialisation of input buffers
    input  = new cl_uchar*[numKernels];
    output = new cl_uchar*[numKernels];

    /// Initialize the 1st prepin buffer with random data. Then copy the data to other prepin buffers
    // 1. Map the 1st buffer
    input[0] = (cl_uchar*)readQueue.enqueueMapBuffer(inPrepinBuffer[0],  CL_TRUE,
               CL_MAP_WRITE_INVALIDATE_REGION, 0, bufferSize, NULL, NULL, &status);
    CHECK_OPENCL_ERROR(status, "enqueueMapBuffer(inPrepinBuffer[0]) failed.");

    // 2. Initialize the buffer with random data.
    for(unsigned int i = 0; i < bufferSize; ++i)
    {
        input[0][i] = (rand() % 256);
    }

    // 3. Initialize the other buffers (copy data from 1st buffer to others)
    for(int k = 1; k < numKernels; ++k)
    {
        input[k] = (cl_uchar*)readQueue.enqueueMapBuffer(inPrepinBuffer[k],  CL_TRUE,
                   CL_MAP_WRITE_INVALIDATE_REGION,
                   0, bufferSize, NULL, NULL, &status);
        CHECK_OPENCL_ERROR(status, "enqueueMapBuffer(inPrepinBuffer) failed.");

        memcpy(input[k], input[0], bufferSize);

        status = readQueue.enqueueUnmapMemObject(inPrepinBuffer[k],  input[k], NULL,
                 NULL);
        CHECK_ERROR(status, SDK_SUCCESS,
                    "Failed enqueueUnmapMemObject(inPrepinBuffer)");
    }
    // 4. Unmap the 1st buffer
    status = readQueue.enqueueUnmapMemObject(inPrepinBuffer[0],  input[0], NULL,
             NULL);
    CHECK_ERROR(status, SDK_SUCCESS,
                "Failed enqueueUnmapMemObject(inPrepinBuffer[0])");

    // Build program from CL file
    SDKFile kernelFile;
    std::string kernelPath = getPath();
    if(sampleArgs-> isLoadBinaryEnabled())
    {
        kernelPath.append(sampleArgs-> loadBinary.c_str());
        if(kernelFile.readBinaryFromFile(kernelPath.c_str()))
        {
            std::cout << "Failed to load kernel file : " << kernelPath << std::endl;
            return SDK_FAILURE;
        }
        cl::Program::Binaries programBinary(1,std::make_pair(
                                                (const void*)kernelFile.source().data(),
                                                kernelFile.source().size()));

        program = cl::Program(context, devices, programBinary, NULL, &status);
        CHECK_OPENCL_ERROR(status, "Program::Program(Binary) failed.");
    }
    else
    {
        kernelPath.append("AsyncDataTransfer_Kernels.cl");
        if(!kernelFile.open(kernelPath.c_str()))
        {
            std::cout << "Failed to load kernel file : " << kernelPath << std::endl;
            return SDK_FAILURE;
        }

        cl::Program::Sources programSource(1,
                                           std::make_pair(kernelFile.source().data(),
                                                   kernelFile.source().size()));

        program = cl::Program(context, programSource, &status);
        CHECK_OPENCL_ERROR(status, "Program::Program(Source) failed.");
    }

    // Get additional options
    std::string flagsStr = std::string("");
    if(sampleArgs->isComplierFlagsSpecified())
    {
        SDKFile flagsFile;
        std::string flagsPath = getPath();
        flagsPath.append(sampleArgs->flags.c_str());
        if(!flagsFile.open(flagsPath.c_str()))
        {
            std::cout << "Failed to load sampleArgs->flags file: " << flagsPath <<
                      std::endl;
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

    status = program.build(devices, flagsStr.c_str());
    if(status != CL_SUCCESS)
    {
        if(status == CL_BUILD_PROGRAM_FAILURE)
        {
            std::string str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>
                              (devices[sampleArgs-> deviceId]);

            std::cout << " \n\t\t\tBUILD LOG\n";
            std::cout << " ************************************************\n";
            std::cout << str << std::endl;
            std::cout << " ************************************************\n";
        }
    }
    CHECK_OPENCL_ERROR(status, "Program::build() failed.");

    // Initialize the kernel object
    kernel = cl::Kernel(program, "work", &status);
    CHECK_OPENCL_ERROR(status, "cl::Kernel(work) failed.");

    globalThreads = bufferSize;
    localThreads = WORKGROUP_SIZE;

    return SDK_SUCCESS;
}

int AsyncDataTransfer::setup()
{
    if (iterations == 0)
    {
        iterations = 1;
    }
    cpuTimer = sampleTimer->createTimer();

    sampleTimer->resetTimer(cpuTimer);
    sampleTimer->startTimer(cpuTimer);
    if(setupCL() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }
    sampleTimer-> stopTimer(cpuTimer);
    setupTime = sampleTimer-> readTimer(cpuTimer) * 1000;

    return SDK_SUCCESS;
}

int AsyncDataTransfer::executeSyncKernel(void)
{
    int status = SDK_SUCCESS;
    writeQueue.finish();
    kernelQueue.finish();
    readQueue.finish();

    sampleTimer->resetTimer(cpuTimer);
    sampleTimer->startTimer(cpuTimer);
    for(int i = 0; i < iterations; i++)
    {
        for(int k = 0; k < numKernels; ++k)
        {
            // Write Buffer
            status |= writeQueue.enqueueWriteBuffer(inputBuffer[k], CL_TRUE, 0, bufferSize,
                                                    input[k], NULL, NULL);

            // Execute Kernel
            status |= kernel.setArg(0, inputBuffer[k]);
            status |= kernel.setArg(1, outputBuffer[k]);
            status |= kernel.setArg(2, k);
            status |= kernelQueue.enqueueNDRangeKernel(kernel, cl::NullRange, globalThreads,
                      localThreads, NULL, NULL);
            status |= kernelQueue.flush();
            status |= kernelQueue.finish();

            // Read Buffer
            status |= readQueue.enqueueReadBuffer(outputBuffer[k], CL_TRUE, 0, bufferSize,
                                                  output[k], NULL, NULL);
        }
    }
    sampleTimer-> stopTimer(cpuTimer);
    syncTime = (sampleTimer-> readTimer(cpuTimer) * 1000) /
               (numKernels * iterations);

    return status;
}


int AsyncDataTransfer::executeAsyncKernel(void)
{
    int status = SDK_SUCCESS;
    std::vector<cl::Event> eventList;
    eventList.reserve(2);
    cl::Event *writeEvents = new cl::Event[iterations * numKernels];
    cl::Event *kernelEvents = new cl::Event[iterations * numKernels];
    cl::Event *readEvents = new cl::Event[iterations * numKernels];
    writeQueue.finish();
    kernelQueue.finish();
    readQueue.finish();

    sampleTimer->resetTimer(cpuTimer);
    sampleTimer->startTimer(cpuTimer);
    for(int i = 0; i < iterations; i++)
    {
        for(int k = 0; k < numKernels; ++k)
        {
            // Write
            eventList.clear();
            if(i > 0)
            {
                eventList.push_back(kernelEvents[((i-1) * numKernels) + k]);
            }
            status |= writeQueue.enqueueWriteBuffer(inputBuffer[k], CL_FALSE, 0, bufferSize,
                                                    input[k],
                                                    &eventList, &(writeEvents[(i * numKernels) + k]));

            // Execute Kernel
            eventList.clear();
            if(i > 0)
            {
                eventList.push_back(readEvents[((i-1) * numKernels) + k]);
            }
            eventList.push_back(writeEvents[(i * numKernels) + k]);
            status |= kernel.setArg(0, inputBuffer[k]);
            status |= kernel.setArg(1, outputBuffer[k]);
            status |= kernel.setArg(2, k);
            status |= kernelQueue.enqueueNDRangeKernel(kernel, cl::NullRange, globalThreads,
                      localThreads,
                      &eventList, &kernelEvents[(i * numKernels) + k]);

            // Read
            eventList.clear();
            eventList.push_back(kernelEvents[(i * numKernels) + k]);
            status |= readQueue.enqueueReadBuffer(outputBuffer[k], CL_FALSE, 0, bufferSize,
                                                  output[k],
                                                  &eventList, &(readEvents[(i * numKernels) + k]));
        }
    }
    // Flush all the queues
    status |= writeQueue.flush();
    status |= kernelQueue.flush();
    status |= readQueue.flush();

    // Wait for finish all the operations
    status |= writeQueue.finish();
    status |= kernelQueue.finish();
    status |= readQueue.finish();

    sampleTimer-> stopTimer(cpuTimer);
    asyncTime = (sampleTimer-> readTimer(cpuTimer) * 1000) /
                (numKernels * iterations);

    // Free events
    delete [] writeEvents;
    delete [] kernelEvents;
    delete [] readEvents;

    return status;
}

int AsyncDataTransfer::runCLKernels(void)
{
    std::cout << "\n Running SyncKernel Version  ..." << std::endl;
    CHECK_ERROR(executeSyncKernel(), SDK_SUCCESS, "Failed executeSyncKernel");
    if(sampleArgs->verify)
    {
        std::cout << " SyncKernel verification  : ";
    }
    if(verifyResults() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    std::cout << "\n Running ASyncKernel Version  ..." << std::endl;
    CHECK_ERROR(executeAsyncKernel(), SDK_SUCCESS, "Failed executeAsyncKernel");
    if(sampleArgs->verify)
    {
        std::cout << " AsyncKernel verification : ";
    }
    if(verifyResults() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    return SDK_SUCCESS;
}

int AsyncDataTransfer::run()
{
    int status;
    // Map in/out prepin buffers
    for(int k = 0; k < numKernels; ++k)
    {
        input[k] = (cl_uchar*)readQueue.enqueueMapBuffer(inPrepinBuffer[k],  CL_TRUE,
                   CL_MAP_WRITE_INVALIDATE_REGION,
                   0, bufferSize, NULL, NULL, &status);
        CHECK_OPENCL_ERROR(status, "enqueueMapBuffer(inPrepinBuffer) failed.");

        output[k] = (cl_uchar*)readQueue.enqueueMapBuffer(outPrepinBuffer[k],  CL_TRUE,
                    CL_MAP_WRITE_INVALIDATE_REGION,
                    0, bufferSize, NULL, NULL, &status);
        CHECK_OPENCL_ERROR(status, "enqueueMapBuffer(inPrepinBuffer) failed.");
    }

    // Warm up
    if (executeSyncKernel() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    std::cout << "Executing kernel for " << iterations
              << " iterations" << std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    // Arguments are set and execution call is enqueued on command buffer
    if (runCLKernels() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // Unmap in/out prepin buffers
    for(int k = 0; k < numKernels; ++k)
    {
        status |= readQueue.enqueueUnmapMemObject(inPrepinBuffer[k], input[k], NULL,
                  NULL);
        status |= readQueue.enqueueUnmapMemObject(outPrepinBuffer[k], output[k], NULL,
                  NULL);
        CHECK_ERROR(status, SDK_SUCCESS, "Failed enqueueUnmapMemObject(PrepinBuffers)");
    }

    return SDK_SUCCESS;
}

int AsyncDataTransfer::cpuReferenceImpl(int numBuffer)
{
    for(size_t i = 0; i < bufferSize; i ++)
    {
        verificationOutput[i] = (input[numBuffer][i] * numBuffer) % 256;
    }

    return SDK_SUCCESS;
}

int AsyncDataTransfer::verifyResults()
{
    if(sampleArgs->verify)
    {
        verificationOutput = (cl_uchar *)malloc(bufferSize);
        CHECK_ALLOCATION(verificationOutput,
                         "Failed to allocate host memory. (verificationOutput)");

        for(int k = 0; k < numKernels; ++k)
        {
            // CPU reference implementation
            CHECK_ERROR(cpuReferenceImpl(k), SDK_SUCCESS, "Failed cpuReferenceImpl");

            //sampleArgs->verify
            for(size_t i = 0; i < bufferSize; ++i)
            {
                if(verificationOutput[i] != output[k][i])
                {
                    std::cout << "Failed! on outputBuffer[" << k << "]\n" << std::endl;
                    return SDK_FAILURE;
                }
            }
        }
        std::cout << "Passed!\n" << std::endl;
    }

    return SDK_SUCCESS;
}

void AsyncDataTransfer::printStats()
{
    if(sampleArgs->timing)
    {
        std::string strArray[4] =
        {
            "Size(Bytes)",
            "Setup Time(ms)",
            "Avg. Sync time(ms)",
            "Avg. Async time(ms)"
        };
        std::string stats[4];

        stats[0] = toString(bufferSize, std::dec);
        stats[1] = toString(setupTime, std::dec);
        stats[2] = toString(syncTime, std::dec);
        stats[3] = toString(asyncTime, std::dec);
        printStatistics(strArray, stats, 4);
    }
}
int AsyncDataTransfer::cleanup()
{
    devices.clear();

    FREE(verificationOutput);

    delete [] input;
    delete [] output;

    delete [] inputBuffer;
    delete [] outputBuffer;
    delete [] inPrepinBuffer;
    delete [] outPrepinBuffer;

    return SDK_SUCCESS;
}

int AsyncDataTransfer::setDeviceInfo(cl::Device& device)
{
    cl_int status = CL_SUCCESS;

    //Get device type
    status = device.getInfo(CL_DEVICE_TYPE, &this->deviceInfo.dType);
    CHECK_OPENCL_ERROR(status, "clGetDeviceIDs(CL_DEVICE_TYPE) failed");

    //Get vender ID
    status = device.getInfo(CL_DEVICE_VENDOR_ID,&this->deviceInfo.venderId);
    CHECK_OPENCL_ERROR(status, "clGetDeviceIDs(CL_DEVICE_VENDOR_ID) failed");

    //Get max compute units
    status = device.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS,
                            &this->deviceInfo.maxComputeUnits);
    CHECK_OPENCL_ERROR(status,
                       "clGetDeviceIDs(CL_DEVICE_MAX_COMPUTE_UNITS) failed");

    //Get max work item dimensions
    status = device.getInfo(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                            &this->deviceInfo.maxWorkItemDims);
    CHECK_OPENCL_ERROR(status,
                       "clGetDeviceIDs(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS) failed");

    // Maximum work group size
    status = device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE,
                            &this->deviceInfo.maxWorkGroupSize);
    CHECK_OPENCL_ERROR(status,
                       "clGetDeviceIDs(CL_DEVICE_MAX_WORK_GROUP_SIZE) failed");

    // Preferred vector sizes of all data types
    status = device.getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR,
                            &this->deviceInfo.preferredCharVecWidth);
    CHECK_OPENCL_ERROR(status,
                       "clGetDeviceIDs(CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR) failed");

    status = device.getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT,
                            &this->deviceInfo.preferredShortVecWidth);
    CHECK_OPENCL_ERROR(status,
                       "clGetDeviceIDs(CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT) failed");

    status = device.getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT,
                            &this->deviceInfo.preferredIntVecWidth);
    CHECK_OPENCL_ERROR(status,
                       "clGetDeviceIDs(CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT) failed");

    status = device.getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG,
                            &this->deviceInfo.preferredLongVecWidth);
    CHECK_OPENCL_ERROR(status,
                       "clGetDeviceIDs(CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG) failed");

    status = device.getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,
                            &this->deviceInfo.preferredFloatVecWidth);
    CHECK_OPENCL_ERROR(status,
                       "clGetDeviceIDs(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT) failed");

    status = device.getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,
                            &this->deviceInfo.preferredDoubleVecWidth);
    CHECK_OPENCL_ERROR(status,
                       "clGetDeviceIDs(CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE) failed");

    status = device.getInfo(CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF,
                            &this->deviceInfo.preferredHalfVecWidth);
    CHECK_OPENCL_ERROR(status,
                       "clGetDeviceIDs(CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF) failed");

    // Clock frequency
    status = device.getInfo(CL_DEVICE_MAX_CLOCK_FREQUENCY,
                            &this->deviceInfo.maxClockFrequency);
    CHECK_OPENCL_ERROR(status,
                       "clGetDeviceIDs(CL_DEVICE_MAX_CLOCK_FREQUENCY) failed");

    // Address bits
    status = device.getInfo(CL_DEVICE_ADDRESS_BITS,&this->deviceInfo.addressBits);
    CHECK_OPENCL_ERROR(status, "clGetDeviceIDs(CL_DEVICE_ADDRESS_BITS) failed");

    // Maximum memory alloc size
    status = device.getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                            &this->deviceInfo.maxMemAllocSize);
    CHECK_OPENCL_ERROR(status,
                       "clGetDeviceIDs(CL_DEVICE_MAX_MEM_ALLOC_SIZE) failed");

    return SDK_SUCCESS;
}

int main(int argc, char * argv[])
{
    AsyncDataTransfer adt;
    if(adt.initialize() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(adt.sampleArgs->parseCommandLine(argc, argv) != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(adt.sampleArgs->isDumpBinaryEnabled())
    {
        return adt.genBinaryImage();
    }

    if(adt.setup() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    if(adt.run() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    adt.printStats();

    if(adt.cleanup() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    return SDK_SUCCESS;
}
