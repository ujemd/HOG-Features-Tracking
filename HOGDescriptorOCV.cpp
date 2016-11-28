//////////////////////////////////////////////////////////////////////////
// Creates C++ MEX-file for extraction of Histogram of Oriented Gradients 
// Features Algorithm in OpenCV. This uses HOGDescriptor class in OpenCV.
//////////////////////////////////////////////////////////////////////////

#include "opencvmex.hpp"
// On some platforms, the following include is needed for "placement new".
// For more information see: http://en.wikipedia.org/wiki/Placement_syntax
#include <memory> 
//#include "objdetect.hpp"

using namespace cv;

static HOGDescriptor *hog = NULL;

//////////////////////////////////////////////////////////////////////////////
// Check inputs
//////////////////////////////////////////////////////////////////////////////
void checkInputs(int nrhs, const mxArray *prhs[])
{
    if ((nrhs < 1) || (nrhs > 2))
    {
        mexErrMsgTxt("Incorrect number of inputs. Function expects 1 or 2 inputs.");
    }
    
//     const int * imDims;
//             
//     // Check input dimensions
//     imDims = mxGetDimensions(prhs[0]);
//     
//     if (mxGetNumberOfDimensions(prhs[0])>2)
//     {
//         mexErrMsgTxt("Incorrect number of dimensions. First input must be a matrix.");
//     }
//     
//     // Check image data type
//     if (!mxIsUint8(prhs[0]))
//     {
//         mexErrMsgTxt("Image must be UINT8.");
//     }
}

//////////////////////////////////////////////////////////////////////////////
// Get MEX function inputs
//////////////////////////////////////////////////////////////////////////////
void getParams(Size &winSize, Size &cellSize, Size &blockSize, Size &blockStride, int &numBins, const mxArray* mxParams)
{
    const mxArray* mxfield;
    int *winSizeInt = NULL,
        *cellSizeInt = NULL,
        *blockSizeInt = NULL,
        *blockOverlap = NULL;
    
    //--winSize--
    mxfield = mxGetField(mxParams, 0, "WindowSize");
    if (mxfield)
        winSizeInt = (int*)mxGetData(mxfield);
    winSize = Size(winSizeInt[1], winSizeInt[0]);

    //--cellSize--
    mxfield = mxGetField(mxParams, 0, "CellSize");
    if (mxfield)
        cellSizeInt = (int*)mxGetData(mxfield);
    cellSize = Size(cellSizeInt[1], cellSizeInt[0]);

    //--blockSize--
    mxfield = mxGetField(mxParams, 0, "BlockSize");
    if (mxfield)
        blockSizeInt = (int*)mxGetData(mxfield);
    blockSize = Size(blockSizeInt[1]*cellSizeInt[1], blockSizeInt[0]*cellSizeInt[0]);
    
    //--blockOverlap--
    mxfield = mxGetField(mxParams, 0, "BlockOverlap");
    if (mxfield)
        blockOverlap = (int*)mxGetData(mxfield);
    blockStride = Size(blockOverlap[1]*cellSizeInt[1],
                       blockOverlap[0]*cellSizeInt[0]);
    
    //--numBins--
    mxfield = mxGetField(mxParams, 0, "NumBins");
    if (mxfield)
        numBins = (int)mxGetScalar(mxfield);
}

//////////////////////////////////////////////////////////////////////////////
// Exit function for freeing persistent memory
//////////////////////////////////////////////////////////////////////////////
void exitFcn() 
{
    if (hog != NULL){
        // explicitly call destructor for "placement new"
        hog->~HOGDescriptor();
        mxFree(hog);
        hog = NULL;
    }
}

//////////////////////////////////////////////////////////////////////////////
// Construct object
//////////////////////////////////////////////////////////////////////////////
void constructObject(const mxArray *prhs[])
{  
    Size winSize,
         cellSize,
         blockSize,
         blockStride;
    int numBins;

    // second input must be struct
    if (mxIsStruct(prhs[1]))
        getParams(winSize, cellSize, blockSize, blockStride, numBins, prhs[1]);

    // Allocate memory for HOG Descriptor model
    hog = (HOGDescriptor *)mxCalloc(1, sizeof(HOGDescriptor));
    // Make memory allocated by MATLAB software persist after MEX-function completes. 
    // This lets us use the updated HOG Descriptor model for the next frame.
    mexMakeMemoryPersistent(hog);
    // Use "placement new" to construct an object on memory that was
    // already allocated using mxCalloc
    new (hog) HOGDescriptor(winSize, blockSize, blockStride, cellSize, numBins, 1, 8);
    
    // Register a function that gets called when the MEX-function is cleared. 
    // This function is responsible for freeing persistent memory
    mexAtExit(exitFcn);
}

//////////////////////////////////////////////////////////////////////////////
// Check compute inputs
//////////////////////////////////////////////////////////////////////////////
void checkComputeInputs(int nrhs, const mxArray *prhs[])
{    
    const int * imDims;
            
    // Check input image dimensions
    imDims = mxGetDimensions(prhs[1]);
    
    if (mxGetNumberOfDimensions(prhs[1])>2)
    {
        mexErrMsgTxt("Incorrect number of dimensions. Second input must be a matrix.");
    }
    
    // Check image data type
    if (!mxIsUint8(prhs[0]))
    {
        mexErrMsgTxt("Image must be UINT8.");
    }
}

//////////////////////////////////////////////////////////////////////////////
// Get MexArray from Vector of float
//////////////////////////////////////////////////////////////////////////////
mxArray * getMexArray (const vector<float> &v)
{
    // Create mex array of type single (real) with size of  input vector (v)
    size_t num = sizeof(float);
    int vectorLength = v.size();
    mxArray *mx = mxCreateNumericMatrix(1, vectorLength, mxSINGLE_CLASS, mxREAL);
    // Copy each element of input vector (v) to mex array
    memcpy( mxGetData(mx), &v[0], vectorLength*num );
    return mx;
}

//////////////////////////////////////////////////////////////////////////////
// Compute HOG features
//////////////////////////////////////////////////////////////////////////////
void computeHOGFeatures(int nlhs, mxArray *plhs[], const mxArray *prhs[])
{
    if (nlhs != 1)
        mexErrMsgTxt("Incorrect number of outputs, must be 1.");
    
    if (hog!=NULL)
    {
        cv::Ptr<cv::Mat> imgCV;//, resimg(hog->winSize.width, hog->winSize.height, CV_8UC1);
        cv::Mat myImg;//, croppedImg;
        // Calculate resized image size (for computing over cropped image)
        cv::Mat resizedImg(hog->winSize.width, hog->winSize.height, CV_8UC1);
        Size imSize;
        vector<Point> locations;
        int outDim;
        
        // Convert mxArray input into OpenCV types
        imgCV = ocvMxArrayToImage_uint8(prhs[1], true);
        myImg = *imgCV;
        // Resize Image
        cv::resize(myImg, resizedImg, cv::Size(hog->winSize.width, hog->winSize.height));    
        
        // Calculate dimensions of output vector
        outDim = (mwSize)( (size_t) hog->nbins
                 * ( hog->blockSize.width / hog->cellSize.width ) 
                 * ( hog->blockSize.height / hog->cellSize.height )
                 * ( ( hog->winSize.width - hog->blockSize.width )
                 / hog->blockStride.width + 1 ) 
                 * ( ( hog->winSize.height - hog->blockSize.height )
                 / hog->blockStride.height + 1 ) );
        
        // Create output vector
        vector<float> descriptorValues(outDim);
        //mxArray *outMatrix;
        // Compute HOG features
        //hog->compute(myImg, descriptorValues, Size(), Size(), locations);
        hog->compute(resizedImg, descriptorValues, Size(), Size(), locations);
        for (int i = 0; i<outDim; i = i +9)
        {
            std::reverse(&(descriptorValues[i]),&(descriptorValues[i+9]));
        }
        // Return features in MexArray
        plhs[0] = getMexArray(descriptorValues);
    }
}

//////////////////////////////////////////////////////////////////////////////
// The main MEX function entry point
//////////////////////////////////////////////////////////////////////////////
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{  	
    checkInputs(nrhs, prhs);
    const char *str = mxIsChar(prhs[0]) ? mxArrayToString(prhs[0]) : NULL;

    if (str != NULL) 
    {
        if (strcmp (str,"construct") == 0)
            constructObject(prhs);
        else if (strcmp (str,"compute") == 0)
            computeHOGFeatures(nlhs, plhs, prhs);
        else if (strcmp (str,"destroy") == 0)
            exitFcn();

        // Free memory allocated by mxArrayToString
        mxFree((void *)str);
    }
}