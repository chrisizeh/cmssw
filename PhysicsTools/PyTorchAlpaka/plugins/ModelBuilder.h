#ifndef PhysicsTools_PyTorchAlpaka_plugins_ModelBuilder_h
#define PhysicsTools_PyTorchAlpaka_plugins_ModelBuilder_h

#include <filesystem>
#include <memory>
#include <array>
#include <cstdlib>
#include <iostream>
#include <boost/filesystem.hpp>

namespace torchtest {

  std::string cmsswPath(std::string path) {
    if (path.size() > 0 && path.substr(0, 1) != "/")
      path = "/" + path;

    std::string base = std::string(std::getenv("CMSSW_BASE"));
    std::string rel = std::string(std::getenv("CMSSW_RELEASE_BASE"));
    return (std::filesystem::exists(base.c_str()) ? base : rel) + path;
  }

  class ModelBuilder {
  public:
    ModelBuilder(std::string path) : script_path_{path} {};

    void setUp() {
      auto path = "/python/" + std::string(std::getenv("SCRAM_ARCH")) + "/" + boost::filesystem::unique_path().string();
      model_path_ = cmsswPath(path);

      std::string cmd = buildCmd();
      std::array<char, 128> buffer;
      std::string result;
      std::shared_ptr<FILE> pipe(popen(cmd.c_str(), "r"), pclose);
      if (!pipe) {
        throw std::runtime_error("Failed to run apptainer to prepare the PyTorch test model: " + cmd);
      }
      while (!feof(pipe.get())) {
        if (fgets(buffer.data(), 128, pipe.get()) != NULL) {
          result += buffer.data();
        }
      }
    }

    void tearDown() {
      if (std::filesystem::exists(model_path_)) {
        std::filesystem::remove_all(model_path_);
      }
    }

    const std::string& script() const { return script_path_; };

    const std::string& modelPath() const { return model_path_; }

  private:
    std::string script_path_;
    std::string model_path_;
    std::string image_ = "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmsml/cmsml:3.11";
    std::string python_exe_ = "python";
    std::string apptainer_cmd_ = "apptainer exec -B";

    const std::string buildCmd() const {
      return apptainer_cmd_ + " " + torchtest::cmsswPath("") + " " + image_ + " " + python_exe_ + " " +
             script() + " " + model_path_;
    }
  };
}  // namespace torchtest
#endif  // PhysicsTools_PyTorchAlpaka_plugins_ModelBuilder_h
