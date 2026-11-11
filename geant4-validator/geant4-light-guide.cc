/**
 * Geant4 Optical Photon Cross-Validation Program for Reconstructed Deformed Light Guides
 *
 * Author: Hengne Li <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
 *
 * Overview:
 * 1. Loads the dense cylindrical radius mesh exported from the bicubic B-spline inverse solver.
 * 2. Constructs a watertight closed 3D shell via G4TessellatedSolid.
 * 3. Attaches a polished dielectric_metal G4OpticalSurface to model the inner reflective mirror film.
 * 4. Emits diffuse G4OpticalPhoton bundles at specified entrance (X, Y) coordinates and counts PMT hits.
 */

#include "G4Box.hh"
#include "G4Element.hh"
#include "G4EmStandardPhysics.hh"
#include "G4LogicalSkinSurface.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4MaterialPropertiesTable.hh"
#include "G4OpticalPhysics.hh"
#include "G4OpticalPhoton.hh"
#include "G4OpticalSurface.hh"
#include "G4PVPlacement.hh"
#include "G4ParticleGun.hh"
#include "G4PhysicalConstants.hh"
#include "G4RunManagerFactory.hh"
#include "G4SystemOfUnits.hh"
#include "G4TessellatedSolid.hh"
#include "G4TriangularFacet.hh"
#include "G4Tubs.hh"
#include "G4UserSteppingAction.hh"
#include "G4VModularPhysicsList.hh"
#include "G4VUserActionInitialization.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "Randomize.hh"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {
std::atomic<long> gPmtHitCount{0};
double gSourceXMm = 0.0;
double gSourceYMm = 0.0;
double gCosineExponent = 1.0;
} // namespace

class DeformedLightGuideConstruction : public G4VUserDetectorConstruction {
public:
    DeformedLightGuideConstruction(const std::string& gridFilePath, double reflectivity)
        : fGridFilePath(gridFilePath), fReflectivity(reflectivity) {}

    G4VPhysicalVolume* Construct() override {
        auto* elN = new G4Element("Nitrogen", "N", 7.0, 14.01 * g / mole);
        auto* elO = new G4Element("Oxygen", "O", 8.0, 16.00 * g / mole);
        auto* airMat = new G4Material("Air", 1.290 * mg / cm3, 2);
        airMat->AddElement(elN, 70 * perCent);
        airMat->AddElement(elO, 30 * perCent);

        auto* shellMat = new G4Material("LightGuideShell", 1.20 * g / cm3, 2);
        shellMat->AddElement(elN, 50 * perCent);
        shellMat->AddElement(elO, 50 * perCent);

        // NOTE: Air must carry an explicit refractive index property table so G4OpBoundaryProcess
        // triggers optical surface reflection at the inner mirror boundary.
        G4double photonEnergies[2] = {2.0 * eV, 3.5 * eV};
        G4double airRefractiveIndex[2] = {1.0, 1.0};
        G4double airAbsorptionLength[2] = {100.0 * m, 100.0 * m};
        auto* airMpt = new G4MaterialPropertiesTable();
        airMpt->AddProperty("RINDEX", photonEnergies, airRefractiveIndex, 2);
        airMpt->AddProperty("ABSLENGTH", photonEnergies, airAbsorptionLength, 2);
        airMat->SetMaterialPropertiesTable(airMpt);

        std::ifstream fin(fGridFilePath);
        int numTheta = 0;
        int numZ = 0;
        double coneLengthMm = 100.0;
        fin >> numTheta >> numZ >> coneLengthMm;

        std::vector<std::vector<double>> radiusGrid(numTheta, std::vector<double>(numZ, 8.0));
        for (int i = 0; i < numTheta; ++i) {
            for (int j = 0; j < numZ; ++j) {
                fin >> radiusGrid[i][j];
            }
        }

        auto* worldSolid = new G4Box("WorldSolid", 40.0 * mm, 40.0 * mm, 140.0 * mm);
        auto* worldLV = new G4LogicalVolume(worldSolid, airMat, "WorldLV");
        auto* worldPV = new G4PVPlacement(nullptr, G4ThreeVector(), worldLV, "WorldPV", nullptr, false, 0);

        auto* shellSolid = new G4TessellatedSolid("DeformedLightGuideSolid");
        const double shellWallMm = 2.0;

        auto makeInnerVertex = [&](int i, int j) {
            int ii = (i % numTheta + numTheta) % numTheta;
            double thetaVal = -pi + (2.0 * pi * ii) / numTheta;
            double zVal = (coneLengthMm * j) / (numZ - 1);
            double rVal = radiusGrid[ii][j];
            return G4ThreeVector(rVal * std::cos(thetaVal) * mm, rVal * std::sin(thetaVal) * mm, zVal * mm);
        };
        auto makeOuterVertex = [&](int i, int j) {
            int ii = (i % numTheta + numTheta) % numTheta;
            double thetaVal = -pi + (2.0 * pi * ii) / numTheta;
            double zVal = (coneLengthMm * j) / (numZ - 1);
            double rVal = radiusGrid[ii][j] + shellWallMm;
            return G4ThreeVector(rVal * std::cos(thetaVal) * mm, rVal * std::sin(thetaVal) * mm, zVal * mm);
        };

        for (int i = 0; i < numTheta; ++i) {
            for (int j = 0; j < numZ - 1; ++j) {
                G4ThreeVector p00 = makeInnerVertex(i, j);
                G4ThreeVector p10 = makeInnerVertex(i + 1, j);
                G4ThreeVector p01 = makeInnerVertex(i, j + 1);
                G4ThreeVector p11 = makeInnerVertex(i + 1, j + 1);
                shellSolid->AddFacet(new G4TriangularFacet(p00, p01, p11, ABSOLUTE));
                shellSolid->AddFacet(new G4TriangularFacet(p00, p11, p10, ABSOLUTE));

                G4ThreeVector q00 = makeOuterVertex(i, j);
                G4ThreeVector q10 = makeOuterVertex(i + 1, j);
                G4ThreeVector q01 = makeOuterVertex(i, j + 1);
                G4ThreeVector q11 = makeOuterVertex(i + 1, j + 1);
                shellSolid->AddFacet(new G4TriangularFacet(q00, q10, q11, ABSOLUTE));
                shellSolid->AddFacet(new G4TriangularFacet(q00, q11, q01, ABSOLUTE));
            }

            shellSolid->AddFacet(
                new G4TriangularFacet(makeInnerVertex(i, 0), makeInnerVertex(i + 1, 0), makeOuterVertex(i + 1, 0), ABSOLUTE)
            );
            shellSolid->AddFacet(
                new G4TriangularFacet(makeInnerVertex(i, 0), makeOuterVertex(i + 1, 0), makeOuterVertex(i, 0), ABSOLUTE)
            );

            int lastZ = numZ - 1;
            shellSolid->AddFacet(
                new G4TriangularFacet(
                    makeInnerVertex(i, lastZ), makeOuterVertex(i + 1, lastZ), makeInnerVertex(i + 1, lastZ), ABSOLUTE
                )
            );
            shellSolid->AddFacet(
                new G4TriangularFacet(
                    makeInnerVertex(i, lastZ), makeOuterVertex(i, lastZ), makeOuterVertex(i + 1, lastZ), ABSOLUTE
                )
            );
        }
        shellSolid->SetSolidClosed(true);

        auto* shellLV = new G4LogicalVolume(shellSolid, shellMat, "LightGuideShellLV");
        new G4PVPlacement(nullptr, G4ThreeVector(), shellLV, "LightGuideShellPV", worldLV, false, 0);

        auto* reflectiveFilm = new G4OpticalSurface("ReflectiveMirrorFilm");
        reflectiveFilm->SetType(dielectric_metal);
        reflectiveFilm->SetModel(unified);
        reflectiveFilm->SetFinish(polished);

        G4double reflValues[2] = {fReflectivity, fReflectivity};
        G4double zeroEff[2] = {0.0, 0.0};
        auto* filmMpt = new G4MaterialPropertiesTable();
        filmMpt->AddProperty("REFLECTIVITY", photonEnergies, reflValues, 2);
        filmMpt->AddProperty("EFFICIENCY", photonEnergies, zeroEff, 2);
        reflectiveFilm->SetMaterialPropertiesTable(filmMpt);
        new G4LogicalSkinSurface("MirrorSkinSurface", shellLV, reflectiveFilm);

        // NOTE: Place a sensitive PMT readout disk behind the exit aperture matching the STEP bore radius (8.18 mm).
        auto* pmtSolid = new G4Tubs("PmtWindowSolid", 0.0, 8.18 * mm, 1.0 * mm, 0.0, 2.0 * pi);
        auto* pmtLV = new G4LogicalVolume(pmtSolid, airMat, "PmtWindowLV");
        new G4PVPlacement(nullptr, G4ThreeVector(0.0, 0.0, (coneLengthMm + 1.05) * mm), pmtLV, "PmtWindowPV", worldLV, false, 0);

        return worldPV;
    }

private:
    std::string fGridFilePath;
    double fReflectivity;
};

class DiffusePointSourceAction : public G4VUserPrimaryGeneratorAction {
public:
    DiffusePointSourceAction() {
        fParticleGun = new G4ParticleGun(1);
        fParticleGun->SetParticleDefinition(G4OpticalPhoton::OpticalPhotonDefinition());
        fParticleGun->SetParticleEnergy(2.5 * eV);
    }
    ~DiffusePointSourceAction() override { delete fParticleGun; }

    void GeneratePrimaries(G4Event* event) override {
        double uPhi = G4UniformRand();
        double uCos = G4UniformRand();
        double phi = 2.0 * pi * uPhi;
        double cosTheta = std::pow(std::max(1.0 - uCos, 1e-9), 1.0 / (gCosineExponent + 1.0));
        double sinTheta = std::sqrt(std::max(0.0, 1.0 - cosTheta * cosTheta));

        G4ThreeVector momentumDir(sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta);
        G4ThreeVector polarizationDir = momentumDir.orthogonal().unit();

        fParticleGun->SetParticlePosition(G4ThreeVector(gSourceXMm * mm, gSourceYMm * mm, 0.002 * mm));
        fParticleGun->SetParticleMomentumDirection(momentumDir);
        fParticleGun->SetParticlePolarization(polarizationDir);
        fParticleGun->GeneratePrimaryVertex(event);
    }

private:
    G4ParticleGun* fParticleGun;
};

class PmtReadoutSteppingAction : public G4UserSteppingAction {
public:
    void UserSteppingAction(const G4Step* step) override {
        auto* track = step->GetTrack();
        auto* postPoint = step->GetPostStepPoint();
        if (!postPoint || !postPoint->GetPhysicalVolume()) {
            return;
        }
        const G4String& volumeName = postPoint->GetPhysicalVolume()->GetName();
        if (volumeName == "PmtWindowPV") {
            gPmtHitCount.fetch_add(1, std::memory_order_relaxed);
            track->SetTrackStatus(fStopAndKill);
        } else if (postPoint->GetPosition().z() < -0.05 * mm) {
            track->SetTrackStatus(fStopAndKill);
        }
    }
};

class LightGuideActionInit : public G4VUserActionInitialization {
public:
    void Build() const override {
        SetUserAction(new DiffusePointSourceAction());
        SetUserAction(new PmtReadoutSteppingAction());
    }
};

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: geant4-light-guide <grid.txt> <points.txt> <out.csv> <reflectivity> <cos_exp> [n_photons]"
                  << std::endl;
        return 1;
    }
    const std::string gridPath = argv[1];
    const std::string pointsPath = argv[2];
    const std::string outputCsvPath = argv[3];
    const double reflectivity = std::stod(argv[4]);
    gCosineExponent = std::stod(argv[5]);
    const int photonsPerPoint = (argc >= 7) ? std::stoi(argv[6]) : 15000;

    // NOTE: Run in single-threaded Serial mode per shard process to keep per-process RSS under ~120 MB.
    auto* runManager = G4RunManagerFactory::CreateRunManager(G4RunManagerType::Serial);
    runManager->SetUserInitialization(new DeformedLightGuideConstruction(gridPath, reflectivity));

    auto* physicsList = new G4VModularPhysicsList();
    physicsList->SetVerboseLevel(0);
    // NOTE: In Geant4 11.x, G4OpticalPhysics requires standard EM processes registered first so e-/e+ ProcessManagers exist.
    physicsList->RegisterPhysics(new G4EmStandardPhysics(0));
    physicsList->RegisterPhysics(new G4OpticalPhysics(0));
    runManager->SetUserInitialization(physicsList);
    runManager->SetUserInitialization(new LightGuideActionInit());
    runManager->Initialize();

    std::ifstream ptsIn(pointsPath);
    std::ofstream csvOut(outputCsvPath);
    csvOut << "x_local_mm,y_local_mm,n_photons,pmt_hits,g4_efficiency\n";

    double xVal = 0.0;
    double yVal = 0.0;
    while (ptsIn >> xVal >> yVal) {
        gSourceXMm = xVal;
        gSourceYMm = yVal;
        gPmtHitCount.store(0);
        runManager->BeamOn(photonsPerPoint);
        const long hits = gPmtHitCount.load();
        const double eff = static_cast<double>(hits) / static_cast<double>(photonsPerPoint);
        csvOut << xVal << "," << yVal << "," << photonsPerPoint << "," << hits << "," << eff << "\n";
    }

    delete runManager;
    return 0;
}
