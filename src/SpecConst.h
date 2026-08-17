#pragma once

#include "Mandrill.h"

union SpecConst {
    uint32_t uintValue;
    float floatValue;
    SpecConst(uint32_t v) : uintValue(v)
    {
    }
    SpecConst(float v) : floatValue(v)
    {
    }
};
