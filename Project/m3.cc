#include "ece408net.h"
#include "ece408test.cc"

void inference_only(int batch_size)
{

  std::cout << "Loading fashion-mnist data...";
  MNIST dataset("../data/fmnist-86/");
  dataset.read_test_data(batch_size);
  std::cout << "Done" << std::endl;

  std::cout << "Loading model...";
  Network dnn = createNetwork_GPU();
  std::cout << "Done" << std::endl;

  dnn.forward(dataset.test_data);
  float acc = compute_accuracy(dnn.output(), dataset.test_labels);
  std::cout << std::endl;
  std::cout << "Test Accuracy: " << acc << std::endl;
  std::cout << std::endl;
}

int main(int argc, char *argv[])
{

  run_testcase("../ece408/project/testcases", false);

  int batch_size = 5000;

  if (argc == 2)
  {
    batch_size = atoi(argv[1]);
  }

  std::cout << "Test batch size: " << batch_size << std::endl;
  inference_only(batch_size);

  return 0;
}