//
// Created by HP on 23/05/2024.
//

#include "utils.cuh"

dim3 computeNumWorkGroups(const dim3& workGroupSize, int width, int height)
{
    return dim3((width + workGroupSize.x - 1) / workGroupSize.x, (height + workGroupSize.y - 1) / workGroupSize.y, 1);
}
