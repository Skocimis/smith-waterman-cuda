import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import math

def transformStr(str):
  l = len(str)
  return np.array(list(map(lambda e: ord(e), list(str))), dtype=np.int32).reshape(1,l)

mch = 5
mss = 3
gap = 9

A = transformStr("CAGCCUCGCUUAG")
B = transformStr("AAUGCCAUUGCCGG")

la = A.shape[1]
lb = B.shape[1]

matrix = np.zeros((lb+1, la+1), np.int32)


mod = SourceModule("""
  __global__ void smithWaterman(int* matrix, int* A, int* B, int la, int lb, int match, int miss, int gap){
    int min = (la<lb)?la:lb;
    int max = (la>lb)?la:lb;
    int k = 1;
    int i = threadIdx.x;
    while(k<min+max){
      int dd = (k<min)?k:((k>max)?(lb+la-k):min);
      __syncthreads();
      if(i+1<=dd){
        int x = ((k<=lb)? (1+i) : (k - lb + 1+i));
		    int y = ((k<=lb)? (k-i) : (lb-i));
        //A[x-1], B[y-1] se uporedjuju
        int v = 0;
        int v1 = matrix[(y-1)*(la+1) + x] - gap;
        int v2 = matrix[y*(la+1) + x - 1] - gap;
        int v3 = matrix[(y-1)*(la+1)+x-1] + ((A[x-1] == B[y-1])?match:(-miss));
        if(v1>v) v = v1;
        if(v2>v) v = v2;
        if(v3>v) v = v3;
		    matrix[y*(la+1) + x] = v;
      }
      k++;
    }
  }
""")

matrix_gpu = cuda.mem_alloc(matrix.nbytes)
cuda.memcpy_htod(matrix_gpu, matrix)
A_gpu = cuda.mem_alloc(A.nbytes)
cuda.memcpy_htod(A_gpu, A)
B_gpu = cuda.mem_alloc(B.nbytes)
cuda.memcpy_htod(B_gpu, B)

func = mod.get_function("smithWaterman")

func(matrix_gpu, A_gpu, B_gpu, np.int32(la), np.int32(lb), np.int32(mch), np.int32(mss), np.int32(gap), block = (min(la, lb), 1, 1), grid = (1, 1, 1))

cuda.memcpy_dtoh(matrix, matrix_gpu)

print("Result: \n", matrix)
