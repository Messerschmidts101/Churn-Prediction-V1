import React from 'react'
import Button from '../Components/Button'
import Header1 from '../Components/Header1'
import Footer from '../Components/Footer'
import { Link } from 'react-router-dom'

function Home() {

    return (
        <>
            <main className='container'>
                <div className='d-flex flex-column text-center'>
                    <div>
                        <Header1>Welcome!</Header1>
                        <Link to="/single_customer"><Button theme={'primary'}>Get Started!</Button></Link>
                    </div>
                </div>
            </main>
            <Footer />
        </>
    )
}

export default Home