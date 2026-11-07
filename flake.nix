{
  description = "Live webcam version of ayoships' What TikTok sees, on a fused WebGPU engine";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAll = f: nixpkgs.lib.genAttrs systems (s: f nixpkgs.legacyPackages.${s});
    in {
      devShells = forAll (pkgs: {
        default = pkgs.mkShell { packages = [ pkgs.nodejs pkgs.python3 pkgs.chromium ]; };
      });
    };
}
