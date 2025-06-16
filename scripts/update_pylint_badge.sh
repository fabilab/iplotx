#!/bin/sh

echo "Switching to uv test environment..."
uv sync --group test

echo "Running pylint on iplotx..."
score=$(uv run pylint iplotx | sed -n 's/^Your code has been rated at \([-0-9.]*\)\/.*/\1/p')
echo "Score: ${score}."

echo "Updating pylint badge..."
uvx anybadge --value=${score} --overwrite --file=assets/pylint.svg pylint
echo "Done."
