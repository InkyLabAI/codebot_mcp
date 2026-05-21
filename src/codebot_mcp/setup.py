"""Setup pipeline: parse repository and generate embeddings (replaces Celery tasks)."""

import os
import uuid
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from codebot_mcp.config import settings
from codebot_mcp.db import Base, Repository, Function, FunctionEmbedding, Class, get_db_path
from codebot_mcp.code_parsing import CodeParser, is_venv_dir, resolve_internal_calls
from codebot_mcp.services.embedding_service import embedding_service
from codebot_mcp.utils.bm25_utils import build_search_text
from codebot_mcp.utils.gitignore_utils import load_gitignore, is_dir_ignored

logger = logging.getLogger(__name__)


def build_modules_dict(repo_path: Path, parser: CodeParser, gitignore=None) -> dict:
    """Build mapping of file paths to module names."""
    modules_dict = {}
    init_modules = {}
    repo_str = str(repo_path)

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [
            d for d in dirs
            if d not in parser.exclude_dirs
            and not is_venv_dir(root, d)
            and not is_dir_ignored(gitignore, repo_str, root, d)
        ]

        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                relative_path = str(file_path.relative_to(repo_path))

                module_name = relative_path.replace("/", ".").replace(".py", "")
                is_init = file == '__init__.py'

                parts = module_name.split(".")
                for i in range(len(parts)):
                    for j in range(i + 1, len(parts) + 1):
                        sub_path = '.'.join(parts[i:j])

                        if is_init:
                            init_modules[sub_path] = module_name
                        elif sub_path not in init_modules:
                            modules_dict[sub_path] = module_name

    modules_dict.update(init_modules)
    return modules_dict


def build_package_exports(repo_path: Path, parser: CodeParser, gitignore=None) -> dict:
    """Build mapping of package exports from __init__.py files."""
    import re
    package_exports = {}
    repo_str = str(repo_path)

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [
            d for d in dirs
            if d not in parser.exclude_dirs
            and not is_venv_dir(root, d)
            and not is_dir_ignored(gitignore, repo_str, root, d)
        ]

        if '__init__.py' in files:
            init_path = Path(root) / '__init__.py'
            relative_dir = str(init_path.parent.relative_to(repo_path))

            if relative_dir == '.':
                package_name = ''
            else:
                package_name = relative_dir.replace('/', '.')

            try:
                with open(init_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                pattern = r'from\s+\.(\w+)\s+import\s+(?:\(([^)]+)\)|([^(\n]+))'
                for match in re.finditer(pattern, content, re.DOTALL):
                    submodule = match.group(1)
                    names_str = match.group(2) or match.group(3)

                    names_str = re.sub(r'#[^\n]*', '', names_str)

                    for name_part in names_str.split(','):
                        name_part = name_part.strip()
                        if not name_part:
                            continue

                        if ' as ' in name_part:
                            parts = name_part.split(' as ')
                            original_name = parts[0].strip()
                            alias = parts[1].strip()
                        else:
                            original_name = name_part.strip()
                            alias = original_name

                        if not original_name or not re.match(r'^[a-zA-Z_]\w*$', original_name):
                            continue
                        if not alias or not re.match(r'^[a-zA-Z_]\w*$', alias):
                            continue

                        if package_name:
                            export_key = f"{package_name}.{alias}"
                            real_path = f"{package_name}.{submodule}.{original_name}"
                        else:
                            export_key = alias
                            real_path = f"{submodule}.{original_name}"

                        package_exports[export_key] = real_path

            except Exception:
                pass

    return package_exports


def setup_repository(repo_path: str) -> str:
    """
    Parse a repository and generate embeddings.

    This is the synchronous setup pipeline that replaces the Celery tasks.
    It writes everything to a SQLite database at {repo_path}/.codebot.db.

    Args:
        repo_path: Path to the repository root directory.

    Returns:
        The repository UUID (as string).
    """
    repo_path = os.path.abspath(repo_path)
    if not os.path.isdir(repo_path):
        raise FileNotFoundError(f"Repository path not found: {repo_path}")

    repo_name = os.path.basename(repo_path)
    db_path = get_db_path(repo_path)

    logger.info("Setting up repository: %s", repo_name)
    logger.info("Database: %s", db_path)

    # Create sync engine + tables
    engine = create_engine(f"sqlite:///{db_path}", echo=False)

    # Enable WAL mode
    from sqlalchemy import event
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()

    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    try:
        # Create repository record
        repo_id = str(uuid.uuid4())
        repository = Repository(
            id=repo_id,
            name=repo_name,
            url=repo_path,
        )
        db.add(repository)
        db.commit()

        # Parse
        logger.info("Parsing code...")
        parser = CodeParser()
        repo_path_obj = Path(repo_path)
        gitignore = load_gitignore(repo_path)

        modules_dict = build_modules_dict(repo_path_obj, parser, gitignore)
        package_exports = build_package_exports(repo_path_obj, parser, gitignore)

        functions_list = []
        classes_list = []
        imports_dict = {}
        module_docstrings = {}
        class_docstrings = {}

        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [
                d for d in dirs
                if d not in parser.exclude_dirs
                and not is_venv_dir(root, d)
                and not is_dir_ignored(gitignore, repo_path, root, d)
            ]

            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    relative_path = str(file_path.relative_to(repo_path_obj))

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            source_code = f.read()

                        functions, classes, imports, module_docstring = parser.parse_code(
                            source_code, relative_path, modules_dict
                        )

                        if module_docstring:
                            module_docstrings[relative_path] = module_docstring

                        functions_list.extend(functions)
                        classes_list.extend(classes)
                        imports_dict.update(imports)

                        for cls in classes:
                            if cls.docstring:
                                class_docstrings[cls.name] = cls.docstring

                    except Exception as e:
                        logger.warning("Error parsing %s: %s", file_path, e)
                        continue

        logger.info("Parsed %d functions, %d classes", len(functions_list), len(classes_list))

        # Resolve internal calls
        logger.info("Resolving internal calls...")
        functions_list = resolve_internal_calls(
            functions=functions_list,
            classes=classes_list,
            imports=imports_dict,
            modules_dict=modules_dict,
            package_exports=package_exports,
            max_workers=4,
        )

        # Store functions
        logger.info("Storing functions...")
        for func_chunk in functions_list:
            module_doc = module_docstrings.get(func_chunk.file_path)
            class_doc = class_docstrings.get(func_chunk.class_name) if func_chunk.class_name else None

            search_text = build_search_text({
                'name': func_chunk.name,
                'file_path': func_chunk.file_path,
                'class_name': func_chunk.class_name,
                'parameters': [p.to_dict() for p in func_chunk.parameters] if func_chunk.parameters else [],
                'return_type': func_chunk.return_type,
                'docstring': func_chunk.docstring,
                'class_docstring': class_doc,
                'module_docstring': module_doc,
            })

            function = Function(
                repository_id=repo_id,
                function_id=func_chunk.id,
                name=func_chunk.name,
                file_path=func_chunk.file_path,
                class_name=func_chunk.class_name,
                nested=func_chunk.nested,
                code=func_chunk.code,
                docstring=func_chunk.docstring,
                module_docstring=module_doc,
                class_docstring=class_doc,
                start_line=func_chunk.start_line,
                end_line=func_chunk.end_line,
                parameters=[p.to_dict() for p in func_chunk.parameters] if func_chunk.parameters else None,
                decorators=func_chunk.decorators,
                return_type=func_chunk.return_type,
                calls=func_chunk.calls,
                search_vector=search_text,
            )
            db.add(function)

        # Store classes
        logger.info("Storing classes...")
        for cls_chunk in classes_list:
            cls = Class(
                repository_id=repo_id,
                class_id=cls_chunk.id,
                name=cls_chunk.name,
                file_path=cls_chunk.file_path,
                code=cls_chunk.code,
                docstring=cls_chunk.docstring,
                start_line=cls_chunk.start_line,
                end_line=cls_chunk.end_line,
                decorators=cls_chunk.decorators,
                superclasses=cls_chunk.superclasses,
            )
            db.add(cls)

        repository.total_functions = len(functions_list)
        repository.total_classes = len(classes_list)
        repository.is_parsed = True
        db.commit()

        # Generate embeddings
        logger.info("Generating embeddings...")
        all_functions = db.query(Function).filter(
            Function.repository_id == repo_id
        ).all()

        function_texts = []
        function_objects = []

        for func in all_functions:
            func_data = {
                'name': func.name,
                'file_path': func.file_path,
                'class_name': func.class_name,
                'docstring': func.docstring,
                'module_docstring': func.module_docstring,
                'class_docstring': func.class_docstring,
                'code': func.code,
                'parameters': func.parameters,
                'return_type': func.return_type,
            }
            text = embedding_service.prepare_function_text(func_data)
            function_texts.append(text)
            function_objects.append(func)

        if function_texts:
            batch_size = settings.BATCH_SIZE
            embeddings_created = 0

            for i in range(0, len(function_texts), batch_size):
                batch_texts = function_texts[i:i + batch_size]
                batch_functions = function_objects[i:i + batch_size]

                embeddings = embedding_service.embed_batch(batch_texts)

                for func, embedding in zip(batch_functions, embeddings):
                    emb_array = np.array(embedding, dtype=np.float32)
                    func_embedding = FunctionEmbedding(
                        function_id=func.id,
                        embedding_blob=emb_array.tobytes(),
                        model_name=settings.EMBEDDING_MODEL,
                    )
                    db.add(func_embedding)
                    embeddings_created += 1

                db.commit()
                logger.info(
                    "Embedded %d/%d functions",
                    min(i + batch_size, len(function_texts)),
                    len(function_texts),
                )

            logger.info("Created %d embeddings", embeddings_created)

        repository.is_embedded = True
        db.commit()

        logger.info("Setup complete! Repository ID: %s", repo_id)
        return repo_id

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
