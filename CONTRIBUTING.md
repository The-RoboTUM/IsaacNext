# 📘 CONTRIBUTING.md
**RoboTUM Software Team | Development Collaboration Guidelines**

---

## 🧭 Core Principle

All our development follows one simple rule:

> 💡 **Every new feature = a new branch.**

The `main` branch must always remain stable, buildable, and ready to demonstrate.
Any new feature or modification should start from a new branch created from `main`.

---

## 🚀 1. Branch Naming Rules

All new feature branches follow this format:

```feature/<short-desc>```

**Example:**

```feature/sim2real```


---

## 👥 2. Collaborative Development on the Same Feature

When multiple team members are working on the same feature (e.g., algorithm, GUI, testing), please follow this workflow:

**【1】The team leader creates the main feature branch:**

```bash
git checkout main
git pull origin main
git checkout -b feature/sim2real
git push -u origin feature/sim2real
```

 **【2】Each team member creates their own sub-branch from the feature branch:**

#### Alice
```bash
git checkout feature/sim2real
git checkout -b feature/sim2real/alice
```

#### Bob
```bash
git checkout feature/sim2real
git checkout -b feature/sim2real/bob
```

Branch naming format:
```bash
feature/<short-desc>/<your-name>
```


 **【3】Once development is done, open a Pull Request to the ```feature branch``` (not directly to ```main```).**

## 🧩 3. Pull Request Guidelines

- Title format: \
Use a short and clear description, for example:
```bash
feature(controller): add sim2real calibration module
```

- Each Pull Request should include:

  - Purpose and key changes
  - Whether local testing is completed
  - Whether a code review is needed

- If the feature is still in progress, you can open a Draft Pull Request to keep others informed.

## 🧱 4. Merge Rules

- All code must be merged through Pull Requests.

- Each Pull Request must receive at least one approval from a reviewer.

- All automated checks (build, test, lint) must pass before merging.

- Use Squash and Merge to keep the main branch history clean.

- After merging, delete the remote branch.

## 🔍 5. Example: Sim2Real Feature Development

Goal:

> Implement a Sim2Real module that allows the robot to reproduce simulation control behavior on real hardware.

Team members: Alice, Bob, Pedro

Branch Structure

```bash
main
 └── feature/sim2real
      ├── feature/sim2real/alice
      ├── feature/sim2real/bob
      └── feature/sim2real/pedro
```

__Example Workflow__

【1】Pedro creates the main feature branch:
```bash
git checkout -b feature/sim2real
git push -u origin feature/sim2real
```

【2】Alice creates her sub-branch and starts development:
```bash
git checkout feature/sim2real
git checkout -b feature/sim2real/alice
```

Example commit message:
```bash
feature(controller): implement real-world noise compensation
```

【3】Bob creates his sub-branch:
```bash
git checkout feature/sim2real
git checkout -b feature/sim2real/bob
```

Example commit message:
```bash
feature(gui): add visualization for sim2real performance
```

【4】Alice and Bob each open a Pull Request to the main feature branch:

- Pull Request #12 → target branch: ```feature/sim2real```

- Pull Request #13 → target branch: ```feature/sim2real```


【5】Pedro reviews and merges their work, then merges the feature branch into main:
```bash
feature(sim2real): add simulation-to-reality transfer module
```

Final project structure remains clean and organized:
```bash
controller/
gui/
integration/
tests/
```


## 💬 6. Code Style and Commit Suggestions

- Keep each commit small and meaningful.

- Use clear naming and concise comments.

- Run tests locally (or at least ensure it builds) before committing.

- Regularly run git pull --rebase origin main to stay updated.

- Avoid mixing unrelated changes in a single Pull Request.


## ✅ 7. Summary

> 🪄 One feature, one branch.
If multiple people work on the same feature, each creates their own sub-branch.
All changes are merged through Pull Requests.
