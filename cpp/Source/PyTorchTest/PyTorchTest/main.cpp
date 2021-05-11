#include <iostream>

#include <ATen/ATen.h>
#include <torch/torch.h>

int main() {
	at::Tensor a = at::ones({ 2,2 });

	std::cout << a << std::endl;
	return 0;
}