from typing import List, Tuple, Any


class DSLParser:

    @classmethod
    def parse_sexprs(cls, lines: List[str]) -> List[Tuple]:
        """Parse multiple S-expression lines"""
        try:
            parser = cls()
            results = []
            for line in lines:
                sexp = parser.parse_sexpr(line)
                if sexp:
                    results.append(sexp)
            return results
        except Exception:
            raise RuntimeError("Could not parse s-expressions")

    def parse_sexpr(self, s: str) -> Tuple:
        """Parse a single S-expression"""
        return self.read_from_tokens(self.tokenize(s))

    def tokenize(self, s: str) -> List[str]:
        """Tokenize an S-expression string"""
        result = s.replace('(', ' ( ').replace(')', ' ) ').split()

        # Discard comments
        for i, tk in enumerate(result):
            if tk[0] == ";":
                return result[:i]

        return result

    def read_from_tokens(self, tokens: List[str]) -> Any:
        """Read an expression from a sequence of tokens"""
        if len(tokens) == 0:
            return None

        token = tokens.pop(0)

        if '(' == token:
            L = []
            while tokens[0] != ')':
                L.append(self.read_from_tokens(tokens))
            tokens.pop(0)  # pop off ')'
            return tuple(L)
        elif ')' == token:
            raise SyntaxError('unexpected )')
        else:
            return token
