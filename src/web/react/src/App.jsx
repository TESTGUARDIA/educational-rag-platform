import { useState, useEffect } from 'react';
import { useAuth } from './hooks/useAuth';
import Login from './pages/Login';
import Header from './components/Header';
import Toast from './components/Toast';

// Teacher pages
import Upload      from './pages/teacher/Upload';
import Submissions from './pages/teacher/Submissions';
import Students    from './pages/teacher/Students';
import TeacherAccount from './pages/teacher/Account';

// Student pages
import StudentEvals   from './pages/student/Evals';
import StudentResults from './pages/student/Results';
import StudentCourses from './pages/student/Courses';
import StudentAccount from './pages/student/Account';

export default function App() {
  const { me, login, logout, loading, error } = useAuth();
  const [page, setPage] = useState(me?.role === 'teacher' ? 'upload' : 'evals');
  const role = me?.role;

  // Après connexion, page par défaut selon le rôle
  useEffect(() => {
    if (role) setPage(role === 'teacher' ? 'upload' : 'evals');
  }, [role]);

  if (!me) {
    return (
      <>
        <Login onLogin={login} loading={loading} error={error} />
        <Toast />
      </>
    );
  }

  function renderPage() {
    if (me.role === 'teacher') {
      switch (page) {
        case 'upload':   return <Upload />;
        case 'subs':     return <Submissions />;
        case 'students': return <Students />;
        case 'account':  return <TeacherAccount me={me} />;
        default:         return <Upload />;
      }
    } else {
      switch (page) {
        case 'evals':   return <StudentEvals />;
        case 'results': return <StudentResults />;
        case 'courses': return <StudentCourses />;
        case 'account': return <StudentAccount me={me} />;
        default:        return <StudentEvals />;
      }
    }
  }

  return (
    <>
      <Header me={me} page={page} setPage={setPage} onLogout={logout} />
      <main id="main-content" style={{
        maxWidth: 1200, margin: '0 auto',
        padding: '28px 20px',
      }}>
        {renderPage()}
      </main>
      <Toast />
    </>
  );
}
