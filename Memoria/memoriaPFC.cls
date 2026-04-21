%% memoriaPFC.cls (Plantilla oficial PFG 2024–25 - Universidad de Deusto)

\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{memoriaPFC}[2025/03/26 v0.5 Plantilla adaptada a normativa PFG 2024–25 de la Universidad de Deusto.]

%% ------------------------------
%% Carga de clase base
%% ------------------------------
\DeclareOption{borrador}{\PassOptionsToPackage{draft}{graphicx}}
\ProcessOptions
\LoadClass[10pt,a4paper,oneside,open=any]{scrbook}  % Base KOMA-Script

%% ------------------------------
%% Paquetes esenciales
%% ------------------------------
\RequirePackage[automark,markcase=upper]{scrlayer-scrpage} % Cabeceras y pies
\RequirePackage[T1]{fontenc}     % Codificación de fuente
\RequirePackage[utf8]{inputenc}  % Codificación de entrada
\RequirePackage[spanish,english]{babel}  % Idioma
\RequirePackage{setspace}        % Interlineado
\RequirePackage[default]{sourcesanspro} % Fuente moderna sans-serif
\renewcommand{\familydefault}{\sfdefault} % Fuente por defecto sans-serif
\RequirePackage{url}             % URLs
\RequirePackage{fancybox}        % Recuadros
\RequirePackage[pdftex]{graphicx} % Imágenes
\RequirePackage{ifthen}          % Lógica condicional
\RequirePackage[pdftex]{color}   % Colores
\RequirePackage{fancyvrb}        % Verbose
\RequirePackage{amssymb}         % Símbolos matemáticos
\RequirePackage{multibib}        % Múltiples bibliografías
\RequirePackage{listings}        % Código fuente
\RequirePackage{pdfpages}        % Incluir PDFs
\RequirePackage[colorlinks=true,
                linkcolor=black,
                citecolor=blue,
                urlcolor=blue,
                unicode,
                hypertexnames=false,
                bookmarks=true]{hyperref}
\RequirePackage{bookmark}        % Mejoras para PDF
\RequirePackage{xcolor}          % Más colores

%% ------------------------------
%% Configuración listings (código)
%% ------------------------------
\definecolor{deustoBlue}{cmyk}{1,0.57,0,0.02}
\definecolor{graybg}{gray}{0.97}
\definecolor{commentgray}{gray}{0.4}
\definecolor{stringorange}{rgb}{0.8,0.4,0.1}

\lstset{
  backgroundcolor=\color{graybg},
  basicstyle=\ttfamily\small,
  keywordstyle=\color{deustoBlue}\bfseries,
  commentstyle=\color{commentgray}\itshape,
  stringstyle=\color{stringorange},
  numberstyle=\tiny\color{gray},
  numbers=left,
  numbersep=10pt,
  frame=single,
  framesep=5pt,
  rulecolor=\color{black},
  breaklines=true,
  showstringspaces=false,
  tabsize=2,
  captionpos=b,
  xleftmargin=1em,
  xrightmargin=1em,
  aboveskip=1em,
  belowskip=1em,
  extendedchars=true,
  inputencoding=utf8
}


%% ------------------------------
%% Definición de metadatos y formato general
%% ------------------------------
\makeatletter
\newcommand{\@autorUno}{Nombre del autor}
\newcommand{\@autorDos}{}
\newcommand{\@autores}{\@autorUno}
\newcommand{\autores}[2]{\def\@autorUno{#1}\def\@autorDos{#2}}
\newcommand{\titulacion}[1]{\def\@titulacion{#1}}
\newcommand{\director}[1]{\def\@director{Director: #1}}
\newcommand{\directora}[1]{\def\@director{Directora: #1}}
\gdef\@resumen{}
\gdef\@descriptores{}
\newcommand{\resumen}[1]{\gdef\@resumen{#1}}
\newcommand{\descriptores}[1]{\gdef\@descriptores{#1}}

\AtBeginDocument{
  \ifthenelse{\equal{\@autorDos}{}}{\def\@autores{\@autorUno}}{\def\@autores{\@autorUno{} y \@autorDos}}
  \pagestyle{scrheadings}
  \setstretch{1.5}          % Interlineado 15 pt para cuerpo 10 pt
  \setlength{\parskip}{12pt}  % Espacio entre párrafos
  \setlength{\parindent}{0pt} % Sin sangría
  \hypersetup{
      linktoc=all,
      pdftitle={\@title},
      pdfauthor={\@autores},
      pdfsubject={Proyecto Fin de Grado},
      pdfkeywords={\@descriptores}
  }
}
\makeatother

%% ------------------------------
%% Márgenes y disposición
%% ------------------------------
\RequirePackage[a4paper,
  top=1.0cm,
  headheight=0.2cm,
  headsep=2.4cm,
  bottom=1.2cm,
  footskip=3.3cm,
  inner=2.5cm,
  outer=2.5cm,
  includehead,
  includefoot
]{geometry}

%% ------------------------------
%% Cabecera y pie de página
%% ------------------------------
\clearpairofpagestyles
\ihead{}
\chead{}
\ohead{{\fontsize{8}{10}\selectfont\textit{PROYECTO FIN DE GRADO}}}  % Cabecera a la derecha

\ifoot{}
\cfoot{}
\ofoot{\raisebox{6pt}{{\fontsize{11}{13}\selectfont\pagemark}}}      % Número de página

%% ------------------------------
%% Estructura del documento
%% ------------------------------
\renewcommand{\frontmatter}{%
  \pagestyle{scrheadings}
  \pagenumbering{roman}
}

\renewcommand{\mainmatter}{%
  \cleardoublepage
  \pagestyle{scrheadings}
  \pagenumbering{arabic}
  \setcounter{secnumdepth}{5}
  \setcounter{tocdepth}{3}

  %% Capítulos: 16 pt, mayúsculas, línea inferior
  \RedeclareSectionCommand[
    beforeskip=0pt,
    afterskip=0.5ex,
    font=\sffamily\bfseries\fontsize{16}{18}\selectfont,
    innerskip=0pt
  ]{chapter}
  \addtokomafont{chapter}{\MakeUppercase}
  \renewcommand*{\chapterformat}{\thechapter.\enskip}
  \renewcommand*{\chapterlinesformat}[3]{%
    \hfill##2##3\par
    \vspace*{-3ex}
    \noindent\rule{\textwidth}{1pt}%
  }

  %% Secciones: 14 pt
  \RedeclareSectionCommand[
    beforeskip=1.5ex plus .2ex,
    afterskip=1ex plus .1ex,
    font=\sffamily\bfseries\fontsize{14}{16}\selectfont
  ]{section}
  \addtokomafont{section}{\MakeUppercase}

  %% Subsecciones: 12 pt aprox.
  \RedeclareSectionCommand[
    beforeskip=1.2ex plus .2ex,
    afterskip=0.8ex plus .1ex,
    font=\sffamily\large\bfseries
  ]{subsection}

  \RedeclareSectionCommand[
    beforeskip=1ex,
    afterskip=0.5ex,
    font=\sffamily\normalsize\bfseries
  ]{subsubsection}
}


\renewcommand{\backmatter}{%
  \cleardoublepage
  \pagestyle{scrheadings}
  \RedeclareSectionCommand[
    beforeskip=0pt,
    afterskip=0.5ex,
    font=\sffamily\bfseries\fontsize{16}{18}\selectfont,
    innerskip=0pt
  ]{chapter}
  \addtokomafont{chapter}{\MakeUppercase}
}


%% ------------------------------
%% Comandos personalizados
%% ------------------------------
\newcommand{\resumenPersonalizado}[2]{%
  \cleardoublepage
  \pagenumbering{roman}  % Numeración en romanos
  \pagestyle{headings}   % Cabecera activa
  \vspace*{3cm}
  {\bfseries\Large Resumen\par}
  \vspace{0.5cm}
  #1 % Texto del resumen
  \vspace{2cm}

  {\bfseries\Large Descriptores\par}
  \vspace{0.5cm}
  #2 % Texto de los descriptores

  \cleardoublepage
}

\newcommand{\hacerresumen}{%
  \cleardoublepage\pagebreak
  {\Large\bfseries Resumen\par\vspace{1em}}%
  \noindent\@resumen\par
  \vspace{2em}%
  {\Large\bfseries Descriptores\par\vspace{1em}}%
  \noindent\@descriptores\thispagestyle{plain}%
}


%% ------------------------------
%% Bibliografía (IEEE)
%% ------------------------------
\newcites{otras}{Bibliograf\'ia}
\newcommand{\bibliografia}[1]{%
  \cleardoublepage
  \addcontentsline{toc}{chapter}{Bibliograf\'ia}
  \renewcommand{\bibname}{BIBLIOGRAF\'IA}
  \bibliographystyle{IEEEtran}
  \bibliography{#1}
}

%% ------------------------------
%% Seudocódigo y párrafos
%% ------------------------------
\renewcommand{\lstlistlistingname}{\'Indice de Seudoc\'odigos}
\renewcommand{\lstlistingname}{Seudoc\'odigo}

\RedeclareSectionCommand[
  beforeskip=1ex,
  afterskip=0.5ex,
  font=\sffamily\normalsize\bfseries,
  runin=false,
  indent=0pt,
  level=5
]{paragraph}
\renewcommand*{\paragraphformat}{\theparagraph.\enskip}

\RedeclareSectionCommand[runin=false]{subparagraph}
