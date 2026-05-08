{
  description = "A Nix-flake-based Python development environment";

  inputs.nixpkgs.url = "https://flakehub.com/f/NixOS/nixpkgs/0.1.*.tar.gz";

  outputs =
    { self, nixpkgs }:
    let
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forEachSupportedSystem =
        f:
        nixpkgs.lib.genAttrs supportedSystems (
          system:
          f {
            pkgs = import nixpkgs { inherit system; };
          }
        );
    in
    {
      devShells = forEachSupportedSystem (
        { pkgs }:
        {
          default = pkgs.mkShell {
            LD_LIBRARY_PATH =
              with pkgs;
              nixpkgs.lib.makeLibraryPath [
                stdenv.cc.cc.lib
                libGL
                glib
              ];
            venvDir = ".venv";
            packages =
              with pkgs;
              [
                python314
                uv
              ]
              ++ (with pkgs.python314Packages; [
                pip
                venvShellHook
              ]);
          };
        }
      );
    };
}
