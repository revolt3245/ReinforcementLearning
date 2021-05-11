#include <iostream>
#include <vector>

#include <ATen/ATen.h>
#include <torch/torch.h>
#include <torch/script.h>

int main() {
	torch::jit::script::Module module;

	std::cout << torch::cuda::device_count() << std::endl;

	try {
		module = torch::jit::load("..\\traced_resnet_model.pt");
		if(torch::cuda::is_available())module.to(at::kCUDA);
	}
	catch (const c10::Error& e) {
		std::cerr << "error loading the module" << std::endl;
		return -1;
	}

	std::cout << "ok" << std::endl;

	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(torch::ones({ 1,3,224,224 }));

	at::Tensor output = module.forward(inputs).toTensor();
	std::cout << output.slice(1, 0, 5) << std::endl;

	return 0;
}