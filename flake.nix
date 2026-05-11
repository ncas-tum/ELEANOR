{
  description = "JAX project with uv";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f {
        pkgs = import nixpkgs { 
          inherit system;
          config.allowUnfree = true;
        };
      });
    in
    {
      devShells = forAllSystems ({ pkgs }: 
      let
          isDarwin = pkgs.stdenv.isDarwin;
          isAarch64Darwin = pkgs.stdenv.isAarch64 && isDarwin;
          hasCuda = pkgs.config.cudaSupport or false;
          
          jaxVariant = 
            if isAarch64Darwin then "metal"
            else if hasCuda then "cuda"
            else "cpu";
      in
      {
        default = pkgs.mkShell {
          packages = with pkgs; [
            git
            uv
            cmake
          ] ++ pkgs.lib.optionals (hasCuda) [
              cudaPackages.cudatoolkit
              cudaPackages.cudnn
            ];
          env = {
              JAX_VARIANT = jaxVariant;
            } // pkgs.lib.optionalAttrs (hasCuda) {
              LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
                pkgs.stdenv.cc.cc
                pkgs.cudaPackages.cudatoolkit
                pkgs.cudaPackages.cudnn
              ];
              XLA_FLAGS = "--xla_gpu_cuda_data_dir=${pkgs.cudaPackages.cudatoolkit}";
            };
        };
      });
    };
}