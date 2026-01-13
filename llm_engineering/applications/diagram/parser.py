class Parser:

    @classmethod
    def parse_sexprs(cls, lines: list[str]):
        try:
            parser = cls()
            results = list()
            for l in lines:
                sexp = parser.parse_sexpr(l)
                if sexp:
                    results.append(sexp)
            return results
        except:
            raise RuntimeError("Could not parse s-expressions")

    def parse_sexpr(self, s :str):
        return self.read_from_tokens(self.tokenize(s))

    def tokenize(self, s :str):
        result = s.replace('(',' ( ').replace(')',' ) ').split()

        # discard comments
        for i, tk in enumerate(result):
          # if tk == ";;":
          if tk[0] == ";":
            return result[:i]

        return result

    def read_from_tokens(self, tokens: list[str]):
        "Read an expression from a sequence of tokens."
        if len(tokens) == 0:
          return
        token = tokens.pop(0)
        if '(' == token:
            L = []
            while tokens[0] != ')':
                L.append(self. read_from_tokens(tokens))
            tokens.pop(0) # pop off ')'
            return tuple(L)
        elif ')' == token:
            raise SyntaxError('unexpected )')
        else:
            return token