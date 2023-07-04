
import Nav from './components/Nav'
import Home from './pages/Home'
import SingleCustomer from './pages/SingleCustomer'
import MultipleCustomers from './pages/MultipleCustomers'
import NotFoundPage from './pages/NotFoundPage'
import { Routes, Route } from 'react-router-dom'
import './css/main.css'
import Footer from './components/Footer'

function App() {
    
    return (
        <>
            <Nav theme="nav" />
            <main className='main bg-light'>
                <Routes>
                    <Route path='/' element={ <Home />} />
                    <Route path='/home' element={ <Home />} />
                    <Route path='/single_customer' element={ <SingleCustomer />} />
                    <Route path='/multiple_customer' element={ <MultipleCustomers />} />
                    <Route path='/*' element={ <NotFoundPage /> } />
                </Routes>
            </main>
            <Footer />
        </>
    )
}

export default App
