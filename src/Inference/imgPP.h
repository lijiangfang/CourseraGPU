#include <string>
#include <vector>
#include <tuple>

struct HostImage
{
    unsigned char *pData; // 8-bit RGB pixels from file
    std::string filename;
    int width, height, channels;
};

std::tuple<std::vector<std::string>, int>  getBatchInputs(std::vector<std::string> filenames, float *d_inputBuffer);

