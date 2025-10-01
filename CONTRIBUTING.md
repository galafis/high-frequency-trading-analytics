# Diretrizes de Contribuição

Bem-vindo(a) ao projeto High-Frequency Trading Analytics! Agradecemos o seu interesse em contribuir. Para garantir um processo de colaboração eficiente e manter a qualidade do código, por favor, siga as diretrizes abaixo.

## Como Contribuir

1.  **Faça um Fork do Repositório**: Comece fazendo um fork deste repositório para a sua conta GitHub.

2.  **Clone o Repositório Forkado**: Clone o seu fork para a sua máquina local:

    ```bash
    git clone https://github.com/SEU_USUARIO/high-frequency-trading-analytics.git
    cd high-frequency-trading-analytics
    ```

3.  **Crie uma Branch**: Crie uma nova branch para a sua feature ou correção. Use nomes descritivos para as branches (ex: `feature/nova-estrategia`, `fix/bug-dashboard`).

    ```bash
    git checkout -b feature/nome-da-sua-feature
    ```

4.  **Instale as Dependências**: Certifique-se de ter todas as dependências instaladas:

    ```bash
    pip install -r requirements.txt
    ```

5.  **Desenvolva e Teste**: Implemente suas alterações e **certifique-se de que todos os testes existentes passem e adicione novos testes** para cobrir seu código. Execute os testes com `pytest`:

    ```bash
    pytest
    ```

6.  **Formate o Código**: Utilize `black` e `flake8` para formatar e lintar seu código. Isso ajuda a manter um estilo de código consistente.

    ```bash
    black . --check
    flake8 .
    ```

7.  **Faça Commit das Suas Alterações**: Escreva mensagens de commit claras e concisas. Se o seu commit resolver um problema, referencie-o (ex: `git commit -m 

feat: Adiciona nova estratégia de trading #123
`git commit -m "feat: Adiciona nova estratégia de trading #123"`

8.  **Envie Suas Alterações**: Faça o push da sua branch para o seu fork no GitHub:

    ```bash
    git push origin feature/nome-da-sua-feature
    ```

9.  **Abra um Pull Request (PR)**: Vá para o repositório original no GitHub e abra um Pull Request da sua branch para a branch `main`.

## Padrões de Código

-   **Python**: Siga o PEP 8 para estilo de código. Use `black` para formatação automática.
-   **Testes**: Cada nova funcionalidade ou correção de bug deve ser acompanhada por testes unitários e/ou de integração relevantes.
-   **Documentação**: Mantenha a documentação atualizada, especialmente o `README.md` e os docstrings do código.

## Boas Práticas

-   Mantenha seus Pull Requests focados em uma única funcionalidade ou correção.
-   Escreva mensagens de commit claras e informativas.
-   Seja respeitoso e construtivo em suas interações.

---

# Contribution Guidelines

Welcome to the High-Frequency Trading Analytics project! We appreciate your interest in contributing. To ensure an efficient collaboration process and maintain code quality, please follow the guidelines below.

## How to Contribute

1.  **Fork the Repository**: Start by forking this repository to your GitHub account.

2.  **Clone Your Forked Repository**: Clone your fork to your local machine:

    ```bash
    git clone https://github.com/YOUR_USERNAME/high-frequency-trading-analytics.git
    cd high-frequency-trading-analytics
    ```

3.  **Create a Branch**: Create a new branch for your feature or fix. Use descriptive names for branches (e.g., `feature/new-strategy`, `fix/dashboard-bug`).

    ```bash
    git checkout -b feature/your-feature-name
    ```

4.  **Install Dependencies**: Make sure you have all dependencies installed:

    ```bash
    pip install -r requirements.txt
    ```

5.  **Develop and Test**: Implement your changes and **ensure that all existing tests pass and add new tests** to cover your code. Run tests with `pytest`:

    ```bash
    pytest
    ```

6.  **Format the Code**: Use `black` and `flake8` to format and lint your code. This helps maintain a consistent code style.

    ```bash
    black . --check
    flake8 .
    ```

7.  **Commit Your Changes**: Write clear and concise commit messages. If your commit resolves an issue, reference it (e.g., `git commit -m "feat: Add new trading strategy #123"`)

8.  **Push Your Changes**: Push your branch to your fork on GitHub:

    ```bash
    git push origin feature/your-feature-name
    ```

9.  **Open a Pull Request (PR)**: Go to the original repository on GitHub and open a Pull Request from your branch to the `main` branch.

## Code Standards

-   **Python**: Follow PEP 8 for code style. Use `black` for automatic formatting.
-   **Tests**: Every new feature or bug fix should be accompanied by relevant unit and/or integration tests.
-   **Documentation**: Keep documentation up-to-date, especially `README.md` and code docstrings.

## Best Practices

-   Keep your Pull Requests focused on a single feature or fix.
-   Write clear and informative commit messages.
-   Be respectful and constructive in your interactions.

