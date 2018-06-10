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

#ifndef UNSHARP_MASK_H_
#define UNSHARP_MASK_H_

#undef _VARIADIC_MAX
#define _VARIADIC_MAX 10

#define __CL_ENABLE_EXCEPTIONS

#define SAMPLE_VERSION "AMD-APP-SDK-v2.9.214.1"

#include <CL/cl.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "CLUtil.hpp"
#include "SDKBitMap.hpp"

using namespace appsdk;

#ifdef APIENTRY
#undef APIENTRY
#endif

#include <GL/glut.h>

#define INPUT_IMAGE "UnsharpMask_Input.bmp"
#define OUTPUT_IMAGE "UnsharpMask_Output.bmp"
#define CPU_OUTPUT_IMAGE "UnsharpMask_Output_CPU.bmp"

#define PI 3.14159265358979f
#define CLAMP(X,LOW,HIGH)  max((LOW), min((HIGH), (X)))

using namespace std;

typedef struct _pixel
{
    unsigned char r,g,b,a;
    bool operator==(const struct _pixel &other)
    {
        return (b == other.b && g == other.g && r == other.r && a == other.a);
    }
    bool operator!=(const struct _pixel &other)
    {
        return !operator==(other);
    };
} Pixel;

int imageWidth,imageHeight;
unsigned char
*imageOutputPtr;   /**< Temporary output image to display in GUI */

class UnsharpMask
{

        std::string imageFile;                  /**< imput image file */
        int height;                             /**< height of the image */
        int width;                              /**< width of the image */
        int radius;                             /**< Gaussian Filter radius */
        float sigma;                            /**< sigma = radius/2.0f */
        float threshold;                        /**< Specify the minimum difference between pixels */
        float amount;                           /**< brightness or the darkness factor */
        bool gui;                               /**< gui option enabled/disabled flag */
        int iterations;                         /**< Number of iterations for kernel execution */
        bool dImageBuffer;                      /**< Use the image data types */
        SDKBitMap inputImage;                   /**< Input image is loaded   */
        cl::Image2D
        inputImageObj;              /**< Input image object (OpenCL class) */
        unsigned char* unsharpMaskImage;        /**< Sharpened output image */
        unsigned char* cpuUnsharpMaskImage;     /**< Sharpened output image (CPU ) */
        size_t imageSize;                       /**< Size of the input image */
        cl::size_t<3> origin,
        region;           /**< Starting and ending Pixel positions */
        cl::Buffer gaussian1DBuffer;            /**< Buffer for guassian 1D kernel */
        cl::Buffer gaussian2DBuffer;            /**< Buffer for guassian 2D kernel */
        cl::Image2D sharpenImageObj;            /**< Output image object **/
        unsigned char* ptr;                     /**< temporary host pointer */
        cl::Buffer
        outputBuffer;                /**< Input buffer which holds input image */
        cl::Buffer
        inputBuffer;                 /**< Output buffer which holds the Sharpened image */
        cl::Buffer tmpImageObj;                 /**< Image after 1st pass */
        size_t rowPitch;
        cl_double setupTime;                    /**< time taken to setup OpenCL resources and building kernel */
        cl_double kernelTime;                   /**< time taken to run kernel and read result back */
        cl::Context context;                    /**< Context */
        std::vector<cl::Device> devices;        /**< vector of devices */
        std::vector<cl::Device> device;         /**< device to be used */
        std::vector<cl::Platform> platforms;    /**< vector of platforms */
        cl::CommandQueue commandQueue,queue;    /**< command queue */
        cl::Program program;                    /**< program object */
        cl::Kernel unsharp_mask_pass1;          /**< kernel template for 1st pass */
        cl::Kernel unsharp_mask_pass2;          /**< kernel template for 2nd pass */
        cl::Kernel
        unsharp_mask_filter;         /**< kernel template for single pass without Image Data types */

        SDKTimer    *sampleTimer;      /**< SDKTimer object */

    public:

        CLCommandArgs   *sampleArgs;   /**< CLCommand argument class */

        /**
        * Constructor
        * Initialize member variables
        */
        UnsharpMask()
            : imageFile(INPUT_IMAGE),
              radius(10),
              threshold(50.0f),
              amount(1.2f),
              gui(false),
              iterations(1),
              setupTime(0),
              kernelTime(0),
              ptr(NULL),
              cpuUnsharpMaskImage(NULL),
              unsharpMaskImage(NULL),
              dImageBuffer(false)
        {
            sampleArgs = new CLCommandArgs();
            sampleTimer = new SDKTimer();
            sampleArgs->sampleVerStr = SAMPLE_VERSION;
            /* Set default values for width and height */
            width = 0;
            height = 0;
        }

        ~UnsharpMask()
        {
        }

        /**
        * Allocate image memory and Load bitmap file
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int setupUnsharpMask();

        /**
        * Override from SDKSample, Generate binary image of given kernel
        * and exit application
        */
        int genBinaryImage();

        /**
        * OpenCL related initialisations.
        * Set up Context, Device list, Command Queue, Memory buffers
        * Build CL kernel program executable
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int setupCL();

        /**
        * Set values for kernels' arguments, enqueue calls to the kernels
        * on to the command queue, wait till end of kernel execution.
        * Get kernel start and end time if timing is enabled
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int runCLKernels();

        /**
        * Reference CPU implementation of Binomial Option
        * for performance comparison
        */
        void UnsharpMaskCPUReference();

        /**
        * Override from SDKSample. Print sample stats.
        */
        void printStats();

        /**
        * Override from SDKSample. Initialize
        * command line parser, add custom options
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int initialize();

        /**
        * Override from SDKSample, adjust width and height
        * of execution domain, perform all sample setup
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int setup();

        /**
        * Override from SDKSample
        * Run OpenCL Sobel Filter
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int run();

        /**
        * Override from SDKSample
        * Cleanup memory allocations
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int cleanup();

        /**
        * Override from SDKSample
        * Verify against reference implementation
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int verifyResults();

        /**
        *Funtion to decide whether to display the image or not.
        */
        bool displayImage()
        {
            return gui;
        }

        /**
        *Funtion to display the image.
        */
        void runGUI(int argc , char *argv[]);

    private:

        /**
        * get the pixel (x,y)
        * for the Image -image
        * @return the pixel (x,y)
        */
        inline Pixel getPixel(unsigned char* image, int x, int y, int width);

        /**
        *set the pixel (x,y)
        *image - Image
        */
        inline void setPixel(unsigned char* image, int x, int y, int width, Pixel p);


        /**
        *genereate the gaussian filter for "kernel"
        *radius - size of the filter
        */
        void generateGaussian1D(float sigma, int radius, float* kernel);

        /**
        *genereate the 2D gaussian filter for "kernel"
        *radius - size of the filter
        */
        void generateGaussian2D(float sigma, int radius, float* kernel);

        /**
        *Run UnsharpMask in the cpu.
        *Used for verification purpose.
        */
        void unsharpMask2PassCPU(unsigned char* input, unsigned char* output
                                 , int width, int height
                                 , float*gk, int gkRadius
                                 , float threshold, float amount) ;
        /**
        *Load the input image.
        */
        int loadInputImage();

};

#endif // UNSHARP_MASK_H_
