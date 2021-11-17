# Crystal Electric Field Analysis
## The Goal
Use inelastic neutron scattering (INS) data to calculate the crystal field Hamiltonian for a given compound, confirm the correct Hamiltonian by predicting thermodynamic properties. (Currently Focusing on the Pr4+ ion, exploring LS coupling basis vs the total angular momentum J basis).
## The Proccess
1. Use the simplest model, the cubic model, to find a good starting point for the Stevens' Coefficients.
2. Lift cubic constraints, introduce the coefficients expected for the material's structure, and fit the model to measured INS data.
3. Using fitted coefficients, create and diagonalize a CF Hamiltonian.
4. Predict thermodynamic properties (Magnetization and Susceptibility) and confirm the predictions with experiment.





## Rambling back story
The Crystal Electric Field (Crystal Field) is a low energy phenomena that describes the static electric field created by a ligand environment in a crystal structure. At low temperatures crystal fields lift degeneracies of electron orbital states. Crystal Field Theory began in the 1930s, and it was untouched for a while due to lack of necessary tools needed for experimentation. With recent advances in technology as well as interest in quantum computing, Crystal Field Theory can be tested and the field has been revived. Inleastic neutron scattering is a key tool for my lab and our studies. By using carefully controlled neutron sources and detectors we can observe the Momentum and Frequnecy (Q,W) responses of our system; this brings in a whole host of theory. I am focusing my attention on systems with Lanthanide (Ln) central ions in varying crystalline structures. Lanthanides are often studied because of their large magnetic moments, and the ability to create crystals in which the Ln ion can be considered by its single ion properties, thus making a connection from theory to reality easier. By constructing the ground state crystal field Hamiltonian we can have an accurate description of the low energy physics taking place. The main goal is to be able to create a topologically insulated qubit. But inorder to creaate a smart quantum magnet, we need to understand it. The crystal field can be described in terms of Stevens' Coefficients and Stevens' Operators. The operators are messy and contain electronic orbital integrals; the math has already been done. There's maximally 15 Stevens' Operators that can be used to describe a crystal field; theory uses symmetries to tell us which terms are necessary for which types of crystal structure. 