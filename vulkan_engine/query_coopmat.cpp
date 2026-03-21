#include "prism_vulkan.h"
#include <stdio.h>
#include <vector>

int main() {
    if (volkInitialize() != VK_SUCCESS) { printf("No Vulkan\n"); return 1; }
    
    VkApplicationInfo appInfo = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo instInfo = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instInfo.pApplicationInfo = &appInfo;
    VkInstance instance;
    vkCreateInstance(&instInfo, nullptr, &instance);
    volkLoadInstance(instance);
    
    uint32_t gpuCount = 0;
    vkEnumeratePhysicalDevices(instance, &gpuCount, nullptr);
    std::vector<VkPhysicalDevice> gpus(gpuCount);
    vkEnumeratePhysicalDevices(instance, &gpuCount, gpus.data());
    
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(gpus[0], &props);
    printf("GPU: %s\n\n", props.deviceName);
    
    // Query cooperative matrix properties
    uint32_t count = 0;
    auto fn = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
        vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    if (!fn) { printf("No cooperative matrix support\n"); return 1; }
    
    fn(gpus[0], &count, nullptr);
    printf("Cooperative matrix configs: %d\n\n", count);
    
    std::vector<VkCooperativeMatrixPropertiesKHR> coopProps(count);
    for (auto& p : coopProps) {
        p.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
        p.pNext = nullptr;
    }
    fn(gpus[0], &count, coopProps.data());
    
    const char* compNames[] = {"???","float16","float32","float64","sint8","sint16","sint32","sint64",
                                "uint8","uint16","uint32","uint64"};
    auto compName = [&](VkComponentTypeKHR t) -> const char* {
        if (t <= 11) return compNames[t];
        return "???";
    };
    const char* scopeNames[] = {"???","device","workgroup","subgroup","queuefamily"};
    auto scopeName = [&](VkScopeKHR s) -> const char* {
        if (s <= 4) return scopeNames[s];
        return "???";
    };
    
    for (uint32_t i = 0; i < count; i++) {
        auto& p = coopProps[i];
        printf("[%d] M=%d K=%d N=%d  A=%s B=%s C=%s Result=%s  scope=%s saturate=%d\n",
               i, p.MSize, p.KSize, p.NSize,
               compName(p.AType), compName(p.BType), 
               compName(p.CType), compName(p.ResultType),
               scopeName(p.scope),
               p.saturatingAccumulation);
    }
    
    vkDestroyInstance(instance, nullptr);
    return 0;
}
