import React from 'react'
import Header2 from './Header2'
import { Link } from 'react-router-dom'

function Footer() {
    return (
        <footer className='bg-dark text-center text-white' style={{position:"absolute",bottom:0,width:"100%"}}>
            <div className='container p-4'>
                <section className='mb-4'>
                    <Link to="https://github.com/Messerschmidts101/Churn-Prediction-V1" className="me-4 text-reset">
                        <i className="fab fa-github"></i>
                    </Link>
                </section>
                <section className='mb-4'>
                    <p>
                        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
                    </p>
                </section>
                <section className='container'>
                    <div className='row'>
                        <div className='col'>
                            <Header2>
                                WebDev Team
                            </Header2>
                            <p>Lintag, Juan Carlo</p>
                            <p>Olaguer, Matthew</p>
                            <p>Paano, Julius Angelo</p>
                        </div>
                        <div className='col'>
                            <Header2>
                                Data Science Team
                            </Header2>
                            <p>Alarcon, Chastin</p>
                            <p>David, Adrien Christian</p>
                            <p>Ferrer, Bryan Kristoffer</p>
                            <p>Señorin, Erika Sheen</p>
                        </div>
                    </div>
                </section>
            </div>
        </footer>
    )
}

export default Footer