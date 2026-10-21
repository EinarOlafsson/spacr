## Modellbibliotek

spaCR innehåller en katalog över tränade modeller och hämtar dem vid behov. Öppna <strong>Model Zoo</strong> från startskärmen för att bläddra bland och installera modellerna, eller ange en modellnyckel i en inställningsfil, till exempel ` pathogen_model: toxoplasma_pv_v1 `. Modellen laddas då ned och verifieras med en kontrollsumma första gången den behövs. Varje publicerad post har en SHA-256-kontrollsumma. En post utan kontrollsumma nekas, eftersom en ofullständig eller utbytt modellfil annars inte kan skiljas från den riktiga.
